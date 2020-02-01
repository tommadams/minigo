// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>
#include <functional>
#include <memory>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/str_split.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/clock.h"
#include "absl/time/time.h"
#include "cc/async/poll_thread.h"
#include "cc/async/sharded_executor.h"
#include "cc/async/thread.h"
#include "cc/async/thread_safe_queue.h"
#include "cc/file/path.h"
#include "cc/file/utils.h"
#include "cc/game.h"
#include "cc/game_utils.h"
#include "cc/init.h"
#include "cc/logging.h"
#include "cc/mcts_tree.h"
#include "cc/model/buffered_model.h"
#include "cc/model/loader.h"
#include "cc/platform/utils.h"
#include "cc/random.h"
#include "cc/tf_utils.h"
#include "cc/wtf_saver.h"
#include "cc/zobrist.h"
#include "gflags/gflags.h"
#include "wtf/macros.h"

// Inference flags.
DEFINE_string(eval_device, "",
              "Optional ID of the device to run the eval model on. For TPUs, "
              "pass the gRPC address.");
DEFINE_string(target_device, "",
              "Optional ID of the device to run the target model on. For TPUs, "
              "pass the gRPC address.");
DEFINE_string(eval_model, "", "Path to the eval model.");
DEFINE_string(target_model, "", "Path to the target model.");

// Tree search flags.
DEFINE_int32(num_readouts, 104,
             "Number of readouts to make during tree search for each move.");
DEFINE_int32(virtual_losses, 8,
             "Number of virtual losses when running tree search.");
DEFINE_double(value_init_penalty, 2.0,
              "New children value initialization penalty.\n"
              "Child value = parent's value - penalty * color, clamped to "
              "[-1, 1].  Penalty should be in [0.0, 2.0].\n"
              "0 is init-to-parent, 2.0 is init-to-loss [default].\n"
              "This behaves similiarly to Leela's FPU \"First Play Urgency\".");
DEFINE_int32(restrict_pass_alive_play_threshold, 4,
             "If the opponent has passed at least "
             "restrict_pass_alive_play_threshold pass moves in a row, playing "
             "moves in pass-alive territory of either player is disallowed.");

// Threading flags.
DEFINE_int32(eval_threads, 2,
             "Number of threads to run batches of eval games on. Must be"
             "divisible by two: odd threads play the eval model as black, "
             "even threds play the target model as black.");
DEFINE_int32(parallel_search, 3, "Number of threads to run tree search on.");
DEFINE_int32(parallel_inference, 2, "Number of threads to run inference on.");
DEFINE_int32(concurrent_games_per_thread, 1,
             "Number of games to play concurrently on each eval thread. "
             "Inferences from a thread's concurrent games are batched up and "
             "evaluated together. Increasing concurrent_games_per_thread can "
             "help improve GPU or TPU utilization, especially for small "
             "models.");

// Game flags.
DEFINE_uint64(seed, 0,
              "Random seed. Use default value of 0 to use a time-based seed. "
              "This seed is used to control the moves played, not whether a "
              "game has resignation disabled or is a holdout.");
DEFINE_double(resign_threshold, -1.0, "Resignation threshold");
DEFINE_int32(num_games, 0,
             "Total number of games to play. Must be divisble by "
             "eval_threads.");

// Output flags.
DEFINE_string(sgf_dir, "",
              "Directory to write output SGFs to. If sgf_dir contains the "
              "substring \"$MODEL\", the name of the last models used for "
              "inference when playing a game will be substituded in the "
              "path.");
DEFINE_string(wtf_trace, "/tmp/minigo.wtf-trace",
              "Output path for WTF traces.");
DEFINE_bool(verbose, true, "Whether to log progress.");
DEFINE_int32(output_threads, 1,
             "Number of threads write training examples on.");

namespace minigo {
namespace {

std::string CleanModelName(const std::string& model_name) {
  return absl::StrReplaceAll(model_name, {{":", "_"}, {"/", "_"}, {".", "_"}});
}

// Information required to run a single inference.
struct Inference {
  MctsNode* leaf;
  ModelInput input;
  ModelOutput output;
};

// Holds all the state for a single eval game.
// Each `EvalThread` plays multiple games in parallel, calling
// `SelectLeaves`, `ProcessInferences` and `PlayMove` sequentially.
class EvalGame {
 public:
  struct Options {
    // Number of virtual losses.
    int num_virtual_losses;

    // Number of positions to read normally.
    int num_readouts;

    // If true, perform verbose logging. Usually restricted to just the first
    // `EvalGame` of the first `EvalThread`.
    bool verbose;

    // Disallow playing in pass-alive territory once the number of passes played
    // during a game is at least `restrict_pass_alive_play_threshold`.
    int restrict_pass_alive_play_threshold;
  };

  EvalGame(int game_id, const Options& options, std::unique_ptr<Game> game,
           std::unique_ptr<MctsTree> black_tree,
           std::unique_ptr<MctsTree> white_tree);

  const MctsTree* GetTree(Color color) const {
    return color == Color::kBlack ? black_tree_.get() : white_tree_.get();
  }

  int game_id() const { return game_id_; }
  Game* game() { return game_.get(); }
  const Game* game() const { return game_.get(); }
  const Options& options() const { return options_; }

  // Selects leaves to perform inference on.
  // Returns the number of leaves selected.
  int SelectLeaves(Color color, std::vector<Inference>* inferences);

  // Processes the inferences selected by `SelectedLeaves` that were evaluated
  // by the EvalThread.
  void ProcessInferences(Color color, absl::Span<const Inference> inferences);

  // Plays a move.
  void PlayMove(Color color, const std::string& model_name);

 private:
  MctsTree* GetTree(Color color) {
    return color == Color::kBlack ? black_tree_.get() : white_tree_.get();
  }

  // Returns true if the predicted win rate is below `resign_threshold`.
  bool ShouldResign(Color toPlay) const;

  const Options options_;
  std::unique_ptr<Game> game_;
  std::unique_ptr<MctsTree> black_tree_;
  std::unique_ptr<MctsTree> white_tree_;
  Random rnd_;

  // Number of consecutive passes played by black and white respectively.
  // Used to determine when to disallow playing in pass-alive territory.
  // `num_consecutive_passes_` latches once it reaches
  // `restrict_pass_alive_play_threshold` is not reset to 0 when a non-pass
  // move is played.
  int num_consecutive_passes_[2] = {0, 0};

  const int game_id_;
};

// The main application class.
// Manages multiple EvalThread objects.
// Each EvalThread plays multiple games concurrently, each one is
// represented by a EvalGame.
// The Evaluator also has a OutputThread, which writes the results of completed
// games to disk.
class Evaluator {
 public:
  Evaluator();

  void Run() LOCKS_EXCLUDED(&mutex_);

  std::unique_ptr<EvalGame> StartNewGame(bool verbose) LOCKS_EXCLUDED(&mutex_);

  void EndGame(std::unique_ptr<EvalGame> eval_game) LOCKS_EXCLUDED(&mutex_);

  // Exectutes `fn` on `parallel_search` threads in parallel on a shared
  // `ShardedExecutor`.
  // Concurrent calls to `ExecuteSharded` are executed sequentially, unless
  // `parallel_search == 1`. This blocking property can be used to pipeline
  // CPU tree search and GPU inference.
  void ExecuteSharded(std::function<void(int, int)> fn);

 private:
  void ParseFlags() EXCLUSIVE_LOCKS_REQUIRED(&mutex_);
  std::unique_ptr<BufferedModel> CreateModels(const std::string& path,
                                              const std::string& device);

  mutable absl::Mutex mutex_;
  Random rnd_ GUARDED_BY(&mutex_);
  WinStats eval_win_stats_ GUARDED_BY(&mutex_);
  WinStats target_win_stats_ GUARDED_BY(&mutex_);
  ThreadSafeQueue<std::unique_ptr<EvalGame>> output_queue_;
  ShardedExecutor executor_;

  // TODO(tommadams): rename BufferedModel to ThreadSafeModel.
  std::unique_ptr<BufferedModel> eval_model_;
  std::unique_ptr<BufferedModel> target_model_;

  int next_game_id_ GUARDED_BY(&mutex_) = 1;

  std::unique_ptr<WtfSaver> wtf_saver_;

  EvalGame::Options eval_options_;
  Game::Options game_options_;
  MctsTree::Options tree_options_;
};

// Plays multiple games concurrently using `EvalGame` instances.
class EvalThread : public Thread {
 public:
  EvalThread(int thread_id, int first_game_id, int num_games,
             Evaluator* evaluator, const Game::Options& game_options,
             const MctsTree::Options& tree_options,
             const EvalGame::Options& eval_options, BufferedModel* black_model_,
             BufferedModel* white_model_);

 private:
  void Run() override;

  Model* GetModel(Color color) {
    return color == Color::kBlack ? black_model_ : white_model_;
  }

  // Starts new games playing.
  void StartNewGames();

  // Selects leaves to perform inference on for all currently playing games.
  // The selected leaves are stored in `inferences_` and `inference_spans_`
  // maps contents of `inferences_` back to the `EvalGames` that they
  // came from.
  void SelectLeaves();

  // Runs inference on the leaves selected by `SelectLeaves`.
  // Runs the name of the model that ran the inferences.
  void RunInferences();

  // Calls `EvalGame::ProcessInferences` for all inferences performed.
  void ProcessInferences();

  // Plays moves on all games that have performed sufficient reads.
  void PlayMoves();

  struct TreeSearch {
    // Holds the span of inferences requested for a single `EvalGame`:
    // `pos` and `len` index into the `inferences` array.
    struct InferenceSpan {
      EvalGame* eval_game;
      size_t pos;
      size_t len;
    };

    void Clear() {
      inferences.clear();
      inference_spans.clear();
    }

    std::vector<Inference> inferences;
    std::vector<InferenceSpan> inference_spans;
  };

  Evaluator* evaluator_;
  const int thread_id_;
  BufferedModel* black_model_;
  BufferedModel* white_model_;
  int next_game_id_;
  std::vector<std::unique_ptr<EvalGame>> eval_games_;
  std::vector<TreeSearch> searches_;
  int num_games_remaining_;
  const Game::Options game_options_;
  const MctsTree::Options tree_options_;
  const EvalGame::Options eval_options_;
  Color to_play_ = Color::kBlack;
};

// Writes SGFs and training examples for completed games to disk.
class OutputThread : public Thread {
 public:
  OutputThread(int thread_id,
               ThreadSafeQueue<std::unique_ptr<EvalGame>>* output_queue);

 private:
  void Run() override;

  ThreadSafeQueue<std::unique_ptr<EvalGame>>* output_queue_;
  const std::string sgf_dir_;
};

Evaluator::Evaluator()
    : rnd_(FLAGS_seed, Random::kUniqueStream),
      executor_(FLAGS_parallel_search) {
  absl::MutexLock lock(&mutex_);
  ParseFlags();
}

EvalGame::EvalGame(int game_id, const Options& options,
                   std::unique_ptr<Game> game,
                   std::unique_ptr<MctsTree> black_tree,
                   std::unique_ptr<MctsTree> white_tree)
    : options_(options),
      game_(std::move(game)),
      black_tree_(std::move(black_tree)),
      white_tree_(std::move(white_tree)),
      rnd_(FLAGS_seed, Random::kUniqueStream),
      game_id_(game_id) {}

int EvalGame::SelectLeaves(Color color, std::vector<Inference>* inferences) {
  auto* tree = GetTree(color);

  int num_leaves_queued = 0;
  do {
    auto* leaf = tree->SelectLeaf(true);
    if (leaf == nullptr) {
      break;
    }

    if (leaf->game_over()) {
      float value =
          leaf->position.CalculateScore(game_->options().komi) > 0 ? 1 : -1;
      tree->IncorporateEndGameResult(leaf, value);
      continue;
    }

    inferences->emplace_back();
    auto& inference = inferences->back();
    inference.input.sym = static_cast<symmetry::Symmetry>(
        rnd_.UniformInt(0, symmetry::kNumSymmetries - 1));
    inference.leaf = leaf;
    leaf->GetPositionHistory(&inference.input.position_history);

    tree->AddVirtualLoss(leaf);

    num_leaves_queued += 1;
  } while (num_leaves_queued < options_.num_virtual_losses);

  return num_leaves_queued;
}

void EvalGame::ProcessInferences(Color color,
                                 absl::Span<const Inference> inferences) {
  auto* tree = GetTree(color);
  for (const auto& inference : inferences) {
    tree->IncorporateResults(inference.leaf, inference.output.policy,
                             inference.output.value);
    tree->RevertVirtualLoss(inference.leaf);
  }
}

void EvalGame::PlayMove(Color color, const std::string& model_name) {
  auto* tree = GetTree(color);

  // Handle resignation.
  if (ShouldResign(color)) {
    game_->SetGameOverBecauseOfResign(OtherColor(color));
  } else {
    // Restrict playing in pass-alive territory once the opponent has passed
    // `restrict_pass_alive_play_threshold` times in a row.
    int num_opponent_passes =
        num_consecutive_passes_[color == Color::kBlack ? 1 : 0];
    bool restrict_pass_alive_moves =
        num_opponent_passes >= options_.restrict_pass_alive_play_threshold;

    // TODO(tommadams): remove PickMove method and have clients decide whether
    // to call SoftPickMove or PickMostVisitedMove.
    Coord c = tree->PickMove(&rnd_, restrict_pass_alive_moves);
    if (options_.verbose) {
      const auto& position = tree->root()->position;
      MG_LOG(INFO) << position.ToPrettyString(true);
      MG_LOG(INFO) << "Move: " << position.n()
                   << " Captures X: " << position.num_captures()[0]
                   << " O: " << position.num_captures()[1];
      MG_LOG(INFO) << absl::StreamFormat("Q: %0.5f", tree->root()->Q());
      MG_LOG(INFO) << "Played >> " << color << "[" << c << "]";
    }

    // Update the number of consecutive passes.
    // The number of consecutive passes latches when it hits
    // `restrict_pass_alive_play_threshold`.
    int& num_passes = num_consecutive_passes_[color == Color::kBlack ? 0 : 1];
    if (num_passes < options_.restrict_pass_alive_play_threshold) {
      if (c == Coord::kPass) {
        num_passes += 1;
      } else {
        num_passes = 0;
      }
    }

    game_->AddNonTrainableMove(color, c, tree->root()->position, model_name,
                               tree->root()->Q(), tree->root()->N());
    auto* other_tree = GetTree(OtherColor(color));
    tree->PlayMove(c);
    other_tree->PlayMove(c);

    // If the whole board is pass-alive, play pass moves to end the game.
    if (tree->root()->position.n() >= kMinPassAliveMoves &&
        tree->root()->position.CalculateWholeBoardPassAlive()) {
      while (!tree->is_game_over()) {
        tree->PlayMove(Coord::kPass);
        other_tree->PlayMove(Coord::kPass);
      }
    }

    // TODO(tommadams): move game over logic out of MctsTree and into Game.
    if (tree->is_game_over()) {
      game_->SetGameOverBecauseOfPasses(
          tree->CalculateScore(game_->options().komi));
    }
  }
}

bool EvalGame::ShouldResign(Color toPlay) const {
  const auto* tree = GetTree(toPlay);
  return game_->options().resign_enabled &&
         tree->root()->Q_perspective() < game_->options().resign_threshold;
}

void Evaluator::Run() {
  eval_model_ = CreateModels(FLAGS_eval_model, FLAGS_eval_device);
  target_model_ = CreateModels(FLAGS_target_model, FLAGS_target_device);
  MG_CHECK(eval_model_->name() != target_model_->name());

  // Initialize the eval threads.
  std::vector<std::unique_ptr<EvalThread>> eval_threads;

  MG_CHECK(FLAGS_num_games % FLAGS_eval_threads == 0)
      << "num_games must be divisible by eval_threads";
  int num_games_per_thread = FLAGS_num_games / FLAGS_eval_threads;

  {
    auto* black = eval_model_.get();
    auto* white = target_model_.get();
    absl::MutexLock lock(&mutex_);
    eval_threads.reserve(FLAGS_eval_threads);
    for (int i = 0; i < FLAGS_eval_threads; ++i) {
      eval_threads.push_back(absl::make_unique<EvalThread>(
          i, i * num_games_per_thread, num_games_per_thread, this,
          game_options_, tree_options_, eval_options_, black, white));
      std::swap(black, white);
    }
  }

  // Start the output threads.
  std::vector<std::unique_ptr<OutputThread>> output_threads;
  for (int i = 0; i < FLAGS_output_threads; ++i) {
    output_threads.push_back(
        absl::make_unique<OutputThread>(i, &output_queue_));
  }
  for (auto& t : output_threads) {
    t->Start();
  }

#ifdef WTF_ENABLE
  // Save WTF in the background periodically.
  wtf_saver_ = absl::make_unique<WtfSaver>(FLAGS_wtf_trace, absl::Seconds(5));
#endif  // WTF_ENABLE

  // Run the eval threads.
  for (auto& t : eval_threads) {
    t->Start();
  }
  for (auto& t : eval_threads) {
    t->Join();
  }

  // Stop the output threads by pushing one null game onto the output queue
  // for each thread, causing the treads to exit when they pop them off.
  for (size_t i = 0; i < output_threads.size(); ++i) {
    output_queue_.Push(nullptr);
  }
  for (auto& t : output_threads) {
    t->Join();
  }
  MG_CHECK(output_queue_.empty());

  {
    absl::MutexLock lock(&mutex_);
    MG_LOG(INFO) << FormatWinStatsTable(
        {{eval_model_->name(), eval_win_stats_},
         {target_model_->name(), target_win_stats_}});
  }
}

void Evaluator::EndGame(std::unique_ptr<EvalGame> eval_game) {
  {
    absl::MutexLock lock(&mutex_);
    auto* game = eval_game->game();
    // Get the name of the winning model.
    const auto& winner_name =
        game->result() > 0 ? game->black_name() : game->white_name();
    // From the name, get whether the winner was the eval or target model and
    // update their stats.
    WinStats& stats = winner_name == eval_model_->name() ? eval_win_stats_
                                                         : target_win_stats_;
    stats.Update(*eval_game->game());
  }
  output_queue_.Push(std::move(eval_game));
}

void Evaluator::ExecuteSharded(std::function<void(int, int)> fn) {
  executor_.Execute(std::move(fn));
}

void Evaluator::ParseFlags() {
  MG_CHECK(FLAGS_eval_threads % 2 == 0)
      << "eval_threads must be a multiple of two";
  MG_CHECK(FLAGS_num_games > 0)
      << "num_games must be set if run_forever is false";
  MG_CHECK(!FLAGS_eval_model.empty());
  MG_CHECK(!FLAGS_target_model.empty());

  // Clamp num_concurrent_games_per_thread to avoid a situation where a single
  // thread ends up playing considerably more games than the others.
  auto max_concurrent_games_per_thread =
      (FLAGS_num_games + FLAGS_eval_threads - 1) / FLAGS_eval_threads;
  FLAGS_concurrent_games_per_thread = std::min(
      max_concurrent_games_per_thread, FLAGS_concurrent_games_per_thread);
  game_options_.resign_threshold = FLAGS_resign_threshold;

  tree_options_.value_init_penalty = FLAGS_value_init_penalty;
  tree_options_.soft_pick_enabled = false;

  eval_options_.num_virtual_losses = FLAGS_virtual_losses;
  eval_options_.num_readouts = FLAGS_num_readouts;
  eval_options_.restrict_pass_alive_play_threshold =
      FLAGS_restrict_pass_alive_play_threshold;
}

std::unique_ptr<BufferedModel> Evaluator::CreateModels(
    const std::string& path, const std::string& device) {
  MG_LOG(INFO) << "Loading model " << path;

  auto def = LoadModelDefinition(path);
  auto* factory = GetModelFactory(def, device);

  std::vector<std::unique_ptr<Model>> impls;
  for (int i = 0; i < FLAGS_parallel_inference; ++i) {
    impls.push_back(factory->NewModel(def));
  }

  return absl::make_unique<BufferedModel>(std::move(impls));
}

EvalThread::EvalThread(int thread_id, int first_game_id, int num_games,
                       Evaluator* evaluator, const Game::Options& game_options,
                       const MctsTree::Options& tree_options,
                       const EvalGame::Options& eval_options,
                       BufferedModel* black_model, BufferedModel* white_model)
    : Thread(absl::StrCat("Eval:", thread_id)),
      evaluator_(evaluator),
      thread_id_(thread_id),
      black_model_(black_model),
      white_model_(white_model),
      next_game_id_(first_game_id),
      num_games_remaining_(num_games),
      game_options_(game_options),
      tree_options_(tree_options),
      eval_options_(eval_options) {
  MG_CHECK((num_games % 2) == 0);
  eval_games_.resize(FLAGS_concurrent_games_per_thread);
}

void EvalThread::Run() {
  WTF_THREAD_ENABLE("EvalThread");

  searches_.resize(FLAGS_parallel_search);
  while (!eval_games_.empty()) {
    StartNewGames();
    for (int i = 0; i < eval_options_.num_readouts;
         i += eval_options_.num_virtual_losses) {
      SelectLeaves();
      RunInferences();
      ProcessInferences();
    }
    PlayMoves();
    to_play_ = OtherColor(to_play_);
  }
}

void EvalThread::StartNewGames() {
  WTF_SCOPE0("StartNewGames");

  // Iterate backwards over the array because it simplifies the
  // num_games_remaining_ == 0 case below.
  for (int i = static_cast<int>(eval_games_.size()) - 1; i >= 0; --i) {
    if (eval_games_[i] != nullptr) {
      // The i'th game is still being played: nothing to do.
      continue;
    }

    if (num_games_remaining_ == 0) {
      // There are no more games to play remove the empty i'th slot from the
      // array. To do this without having to shuffle all the elements down,
      // we move the last element into position i and pop off the back.
      eval_games_[i] = std::move(eval_games_.back());
      eval_games_.pop_back();
      continue;
    }

    WTF_SCOPE0("StartNewGame");

    num_games_remaining_ -= 1;
    auto game = absl::make_unique<Game>(black_model_->name(),
                                        white_model_->name(), game_options_);
    auto black_tree =
        absl::make_unique<MctsTree>(Position(Color::kBlack), tree_options_);
    auto white_tree =
        absl::make_unique<MctsTree>(Position(Color::kBlack), tree_options_);

    auto eval_options = eval_options_;
    eval_options.verbose = FLAGS_verbose && thread_id_ == 0 && i == 0;
    eval_games_[i] = absl::make_unique<EvalGame>(
        next_game_id_++, eval_options, std::move(game), std::move(black_tree),
        std::move(white_tree));
  }
}

void EvalThread::SelectLeaves() {
  WTF_SCOPE("SelectLeaves: games", size_t)(eval_games_.size());

  std::atomic<size_t> game_idx(0);
  evaluator_->ExecuteSharded([this, &game_idx](int shard_idx, int num_shards) {
    WTF_SCOPE0("SelectLeaf");
    MG_CHECK(static_cast<size_t>(num_shards) == searches_.size());

    auto& search = searches_[shard_idx];
    search.Clear();

    for (;;) {
      auto i = game_idx.fetch_add(1);
      if (i >= eval_games_.size()) {
        break;
      }

      TreeSearch::InferenceSpan span;
      span.eval_game = eval_games_[i].get();
      span.pos = search.inferences.size();
      span.len = span.eval_game->SelectLeaves(to_play_, &search.inferences);
      if (span.len > 0) {
        search.inference_spans.push_back(span);
      }
    }
  });
}

void EvalThread::RunInferences() {
  WTF_SCOPE0("RunInferences");

  // TODO(tommadams): stop allocating theses temporary vectors.
  std::vector<const ModelInput*> input_ptrs;
  std::vector<ModelOutput*> output_ptrs;
  for (auto& s : searches_) {
    for (auto& x : s.inferences) {
      input_ptrs.push_back(&x.input);
      output_ptrs.push_back(&x.output);
    }
  }

  if (input_ptrs.empty()) {
    return;
  }

  auto* model = GetModel(to_play_);
  model->RunMany(input_ptrs, &output_ptrs, nullptr);
}

void EvalThread::ProcessInferences() {
  WTF_SCOPE0("ProcessInferences");
  for (auto& s : searches_) {
    for (const auto& span : s.inference_spans) {
      span.eval_game->ProcessInferences(
          to_play_, absl::MakeSpan(s.inferences).subspan(span.pos, span.len));
    }
  }
}

void EvalThread::PlayMoves() {
  WTF_SCOPE0("PlayMoves");
  const auto& model_name = GetModel(to_play_)->name();
  for (auto& eval_game : eval_games_) {
    eval_game->PlayMove(to_play_, model_name);
    if (eval_game->game()->game_over()) {
      evaluator_->EndGame(std::move(eval_game));
      eval_game = nullptr;
    }
  }
}

OutputThread::OutputThread(
    int thread_id, ThreadSafeQueue<std::unique_ptr<EvalGame>>* output_queue)
    : Thread(absl::StrCat("Output:", thread_id)),
      output_queue_(output_queue),
      sgf_dir_(FLAGS_sgf_dir) {}

void OutputThread::Run() {
  for (;;) {
    auto eval_game = output_queue_->Pop();
    if (eval_game == nullptr) {
      break;
    }
    if (!sgf_dir_.empty()) {
      auto black_name = CleanModelName(eval_game->game()->black_name());
      auto white_name = CleanModelName(eval_game->game()->white_name());
      auto output_name = absl::StrCat(GetOutputName(eval_game->game_id()), "-",
                                      black_name, "-", white_name);
    }
  }
}

}  // namespace
}  // namespace minigo

int main(int argc, char* argv[]) {
  minigo::Init(&argc, &argv);
  minigo::zobrist::Init(FLAGS_seed);

  {
    minigo::Evaluator evaluator;
    evaluator.Run();
  }

  minigo::ShutdownModelFactories();

  return 0;
}
