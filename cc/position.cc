// Copyright 2018 Google LLC
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

#include "cc/position.h"

#include <algorithm>
#include <array>
#include <sstream>
#include <type_traits>
#include <utility>

#include "absl/strings/str_format.h"
#include "cc/tiny_set.h"

namespace minigo {

namespace {

constexpr char kPrintWhite[] = "\x1b[0;31;47m";
constexpr char kPrintBlack[] = "\x1b[0;31;40m";
constexpr char kPrintEmpty[] = "\x1b[0;31;43m";
constexpr char kPrintNormal[] = "\x1b[0m";

// A simple array-backed map for storing objects at points on the board.
// Used for computing pass-alive regions.
template <typename V>
class PointMap {
 public:
  // Insert a new element into the map at the given coordinate c.
  // c must be >= 0 and < kN * kN.
  // Inserting multiple elements into the map at the same coordinate results in
  // undefined behavior.
  template <typename... Args>
  V& emplace(Coord c, Args&&... args) {
    coords_.push_back(c);
    return *(new (&data_[c]) V(std::forward<Args>(args)...));
  }

  // Access an element that has been inserted into the map at coordinate c.
  // Accessing an element that hasn't been inserted into the map results in
  // undefined behavior.
  V& operator[](Coord c) {
    MG_DCHECK(std::find(coords_.begin(), coords_.end(), c) != coords_.end());
    return *reinterpret_cast<V*>(&data_[c]);
  }
  const V& operator[](Coord c) const {
    MG_DCHECK(std::find(coords_.begin(), coords_.end(), c) != coords_.end());
    return *reinterpret_cast<const V*>(&data_[c]);
  }

  // Return the coordinates of elements that have been inserted into the map.
  const inline_vector<Coord, kN * kN>& coords() const { return coords_; }

 private:
  typename std::aligned_storage<sizeof(V), alignof(V)>::type data_[kN * kN];
  inline_vector<Coord, kN * kN> coords_;
};

// A fixed-capacity stack of Coords used when flood filling points on the board.
class CoordStack : private inline_vector<Coord, kN * kN> {
  using Impl = inline_vector<Coord, kN * kN>;

 public:
  using Impl::empty;

  void push(Coord c) { Impl::push_back(c); }

  Coord pop() {
    auto result = Impl::back();
    Impl::pop_back();
    return result;
  }
};

}  // namespace

const std::array<inline_vector<Coord, 4>, kN* kN> kNeighborCoords = []() {
  std::array<inline_vector<Coord, 4>, kN * kN> result;
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      auto& coords = result[row * kN + col];
      if (col > 0) {
        coords.emplace_back(row, col - 1);
      }
      if (col < kN - 1) {
        coords.emplace_back(row, col + 1);
      }
      if (row > 0) {
        coords.emplace_back(row - 1, col);
      }
      if (row < kN - 1) {
        coords.emplace_back(row + 1, col);
      }
    }
  }
  return result;
}();

zobrist::Hash Position::CalculateStoneHash(
    const std::array<Color, kN * kN>& stones) {
  zobrist::Hash hash = 0;
  for (int c = 0; c < kN * kN; ++c) {
    hash ^= zobrist::MoveHash(c, stones[c]);
  }
  return hash;
}

Position::Position(Color to_play) : to_play_(to_play) {}

Position::UndoState Position::PlayMove(Coord c, Color color,
                                       ZobristHistory* zobrist_history) {
  UndoState undo(c, to_play_, ko_);
  if (c == Coord::kPass || c == Coord::kResign) {
    ko_ = Coord::kInvalid;
  } else {
    if (color == Color::kEmpty) {
      color = to_play_;
    } else {
      to_play_ = color;
    }
    MG_DCHECK(ClassifyMove(c) != MoveType::kIllegal) << c;
    undo.captures = AddStoneToBoard(c, color);
  }

  n_ += 1;
  to_play_ = OtherColor(to_play_);
  UpdateLegalMoves(zobrist_history);

  return undo;
}

void Position::UndoMove(const UndoState& undo,
                        ZobristHistory* zobrist_history) {
  to_play_ = undo.to_play;
  ko_ = undo.ko;
  n_ -= 1;
  Coord undo_c = undo.c;

  if (undo_c != Coord::kPass) {
    auto undo_color = point_color(undo_c);
    MG_CHECK(undo_color != Color::kEmpty);
    auto undo_head = chain_head(undo_c);
    auto undo_next = chain_next(undo_c);
    auto size = points_[undo_head].bits & Point::kSizeBits;

    // Unlink the stone from the chain.
    // We'll update the chain size and liberty count after.
    if (undo_c == undo_head) {
      // Removing the head of a chain.
      if (undo_next != Coord::kInvalid) {
        // The chain is larger than one stone:
        //  - copy the number of liberties & chain size from the old head.
        //  - update the rest of the stones to point to the new head.
        points_[undo_next].liberties_prev = points_[undo_c].liberties_prev;
        points_[undo_next].bits =
            (points_[undo_next].bits & ~Point::kSizeBits) | size |
            Point::kIsHeadBit;

        // Update the chain's head to the next stone in the chain.
        // TODO(tommadams): there's a minor opitimization available here: if
        // removing this stone would split a chain then we can avoid updating
        // all the stones' head references here because we'll just recompute the
        // chain later.
        undo_head = undo_next;
        for (auto chain_c = chain_next(undo_next); chain_c != Coord::kInvalid;
             chain_c = chain_next(chain_c)) {
          points_[chain_c].bits =
              (points_[chain_c].bits & ~Point::kSizeBits) | undo_head;
        }
      }
    } else {
      // Removing a non-head stone: patch next & previous.
      auto undo_prev = points_[undo_c].liberties_prev;
      points_[undo_prev].next = undo_next;
      if (undo_next != Coord::kInvalid) {
        points_[undo_next].liberties_prev = undo_prev;
      }
    }

    // Remove the stone from the board.
    points_[undo_c] = {};
    stone_hash_ ^= zobrist::MoveHash(undo_c, undo_color);

    // Update the liberty counts of neighboring chains and count how many
    // neighboring stones belong to the same chain as the stone removed by the
    // undo. If there are more than one neighbors in the same chain, the chain
    // may have been split by the removal of this stone.
    tiny_set<Coord, 4> neighbor_chains;
    int num_chain_neighbors = 0;
    int num_lost_liberties = 0;
    for (auto nc : kNeighborCoords[undo_c]) {
      if (is_empty(nc)) {
        // A liberty isn't lost if it's also the liberty of another stone in the
        // same chain.
        if (!HasNeighboringChain(nc, undo_head)) {
          num_lost_liberties += 1;
        }
        continue;
      }
      auto nh = chain_head(nc);
      if (nh == undo_head) {
        num_chain_neighbors += 1;
      }
      if (neighbor_chains.insert(nh)) {
        points_[nh].liberties_prev += 1;
      }
    }

    if (num_chain_neighbors > 1) {
      TaggedPointVisitor visitor(Coord::kInvalid);

      // The stone removed by this undo had more than one neighbor of the same
      // color: it's possible that the removal of this stone has split a chain.
      for (auto nc : kNeighborCoords[undo_c]) {
        if (point_color(nc) == undo_color && !visitor.HasAnyVisit(nc)) {
          RebuildChain(nc, &visitor);
        }
      }
    } else {
      if (size > 1) {
        points_[undo_head].bits -= 1;
        points_[undo_head].liberties_prev -= num_lost_liberties;
      }
    }

    // Put any captured stones back on the board, updating their neighbouring
    // chains' liberty counts.
    auto other_color = OtherColor(undo_color);
    for (auto cc : undo.captures) {
      UncaptureChain(other_color, undo_c, cc);
    }
  }

  UpdateLegalMoves(zobrist_history);
}

std::string Position::ToSimpleString() const {
  std::ostringstream oss;
  for (int row = 0; row < kN; ++row) {
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      auto color = point_color(c);
      if (color == Color::kWhite) {
        oss << "O";
      } else if (color == Color::kBlack) {
        oss << "X";
      } else {
        oss << (c == ko_ ? "*" : ".");
      }
    }
    if (row + 1 < kN) {
      oss << "\n";
    }
  }
  return oss.str();
}

std::string Position::ToPrettyString(bool use_ansi_colors) const {
  std::ostringstream oss;

  auto format_cols = [&oss]() {
    oss << "   ";
    for (int i = 0; i < kN; ++i) {
      oss << Coord::kGtpColumns[i] << " ";
    }
  };

  const char* print_white = use_ansi_colors ? kPrintWhite : "";
  const char* print_black = use_ansi_colors ? kPrintBlack : "";
  const char* print_empty = use_ansi_colors ? kPrintEmpty : "";
  const char* print_normal = use_ansi_colors ? kPrintNormal : "";

  format_cols();
  oss << "\n";
  for (int row = 0; row < kN; ++row) {
    oss << absl::StreamFormat("%2d ", kN - row);
    for (int col = 0; col < kN; ++col) {
      Coord c(row, col);
      auto color = point_color(c);
      if (color == Color::kWhite) {
        oss << print_white << "O ";
      } else if (color == Color::kBlack) {
        oss << print_black << "X ";
      } else {
        oss << print_empty << (c == ko_ ? "* " : ". ");
      }
    }
    oss << print_normal << absl::StreamFormat("%2d", kN - row);
    oss << "\n";
  }
  format_cols();
  return oss.str();
}

inline_vector<Coord, 4> Position::AddStoneToBoard(Coord c, Color color) {
  auto potential_ko = IsKoish(c);
  auto opponent_color = OtherColor(color);

  // Traverse the coord's neighbors, building useful information:
  //  - list of captured chains (if any).
  //  - coordinates of the new stone's liberties.
  //  - set of neighboring chains of the player's color.
  inline_vector<Coord, 4> captured_neighbors;
  inline_vector<Coord, 4> liberties;
  tiny_set<Coord, 4> opponent_chains;
  tiny_set<Coord, 4> player_chains;
  for (auto nc : kNeighborCoords[c]) {
    auto neighbor_color = point_color(nc);
    if (neighbor_color == Color::kEmpty) {
      // Remember the coord of this liberty.
      liberties.push_back(nc);
      continue;
    }

    auto nh = chain_head(nc);
    if (neighbor_color == color) {
      // Remember neighboring chains of same color.
      player_chains.insert(nh);
    } else {
      // Decrement neighboring opponent chain liberty counts and remember the
      // chains we will capture. We'll remove them from the board shortly.
      if (opponent_chains.insert(nh)) {
        if (--points_[nh].liberties_prev == 0) {
          captured_neighbors.push_back(nc);
        }
      }
    }
  }

  // Place the new stone on the board.
  auto color_bits = (static_cast<uint16_t>(color) << Point::kColorShift);
  if (player_chains.empty()) {
    // The stone doesn't connect to any neighbors: create a new chain.
    constexpr uint16_t chain_size = 1;
    points_[c].liberties_prev = liberties.size();
    points_[c].next = Coord::kInvalid;
    points_[c].bits = color_bits | Point::kIsHeadBit | chain_size;
  } else {
    // Designate the first chain in `player_chains` the primary chain. All
    // chains that the newly placed stone connect together will be spliced into
    // this primary chain.
    auto primary_head = player_chains[0];

    // If newly placed stone only connects to one existing chain we can update
    // its liberty counts directly. If the new stone connects multiple chains,
    // we'll recompute the liberties from scratch later.
    if (player_chains.size() == 1) {
      int liberty_delta = -1;
      for (auto nc : liberties) {
        if (!HasNeighboringChain(nc, primary_head)) {
          liberty_delta += 1;
        }
      }
      points_[primary_head].liberties_prev += liberty_delta;
    }

    // First, insert the newly placed stone into the primary chain immediately
    // after its head.
    points_[c].liberties_prev = primary_head;
    points_[c].next = points_[primary_head].next;
    points_[c].bits = color_bits | primary_head;
    if (points_[c].next != Coord::kInvalid) {
      points_[points_[c].next].liberties_prev = c;
    }
    points_[primary_head].next = c;
    points_[primary_head].bits += 1;

    // Splice any remaining neighbor chains into the primary chain. The new
    // chains are spliced in before the newly placed stone. On the first
    // iteration through the loop, the splice point is between the primary
    // chain's head and the new stone; on subsequent iterations it is between
    // the previously spliced chain and the newly placed stone.
    for (int i = 1; i < player_chains.size(); ++i) {
      // Splice the head of this chain in after the primary head.
      auto splice_prev = points_[c].liberties_prev;
      auto splice_head = player_chains[i];
      points_[primary_head].bits += chain_size(splice_head);
      points_[splice_prev].next = splice_head;
      points_[splice_head].liberties_prev = splice_prev;

      // Clear the "is head" bit of the spliced chain's head.
      points_[splice_head].bits &= ~Point::kIsHeadBit;

      for (auto splice_c = splice_head;; splice_c = chain_next(splice_c)) {
        // Update the head references for all stones in the spliced chain.
        points_[splice_c].bits =
            (points_[splice_c].bits & ~Point::kSizeBits) | primary_head;
        if (points_[splice_c].next == Coord::kInvalid) {
          // Splice the tail of this chain in before the newly placed stone.
          points_[splice_c].next = c;
          points_[c].liberties_prev = splice_c;
          break;
        }
      }
    }

    // The newly placed stone joined multiple chains, recompute liberties from
    // scratch.
    if (player_chains.size() > 1) {
      OneTimePointVisitor visitor;
      int num_liberties = 0;
      for (auto chain_c = primary_head; chain_c != Coord::kInvalid;
           chain_c = points_[chain_c].next) {
        for (auto nc : kNeighborCoords[chain_c]) {
          if (is_empty(nc) && visitor.Visit(nc)) {
            num_liberties += 1;
          }
        }
      }
      points_[primary_head].liberties_prev = num_liberties;
    }
  }

  stone_hash_ ^= zobrist::MoveHash(c, color);

  // Update ko.
  if (captured_neighbors.size() == 1 && potential_ko == opponent_color &&
      chain_size(captured_neighbors[0]) == 1) {
    ko_ = captured_neighbors[0];
  } else {
    ko_ = Coord::kInvalid;
  }

  // Remove captured chains.
  for (auto nc : captured_neighbors) {
    int num_captured_stones = chain_size(nc);
    if (color == Color::kBlack) {
      num_captures_[0] += num_captured_stones;
    } else {
      num_captures_[1] += num_captured_stones;
    }
    RemoveChain(nc);
  }

#ifndef NDEBUG
  Validate();
#endif

  return captured_neighbors;
}

void Position::RemoveChain(Coord c) {
  auto removed_color = point_color(c);
  auto other_color = OtherColor(removed_color);

  c = chain_head(c);
  for (;;) {
    stone_hash_ ^= zobrist::MoveHash(c, removed_color);
    tiny_set<Coord, 4> other_chains;
    for (auto nc : kNeighborCoords[c]) {
      if (point_color(nc) == other_color) {
        auto nh = chain_head(nc);
        if (other_chains.insert(nh)) {
          points_[nh].liberties_prev += 1;
        }
      }
    }

    // Get the next point in the chain before clearing the current point.
    auto next = chain_next(c);
    points_[c] = {};
    if (next == Coord::kInvalid) {
      break;
    }
    c = next;
  }
}

void Position::UncaptureChain(Color color, Coord capture_c, Coord chain_c) {
  auto other_color = OtherColor(color);
  auto color_bits = static_cast<uint16_t>(color) << Point::kColorShift;

  // Create a new chain whose head is at `chain_c`.
  auto head = chain_c;
  CoordStack coord_stack;
  coord_stack.push(head);
  points_[head].bits = color_bits;

  auto prev = chain_c;
  int size = 0;
  while (!coord_stack.empty()) {
    auto c = coord_stack.pop();
    stone_hash_ ^= zobrist::MoveHash(c, color);
    size += 1;

    // Patch the chain's linked list references.
    points_[c].liberties_prev = prev;
    points_[prev].next = c;
    prev = c;

    tiny_set<Coord, 4> neighbor_chains;
    for (auto nc : kNeighborCoords[c]) {
      if (nc != capture_c) {
        auto neighbor_color = points_[nc].color();
        if (neighbor_color == Color::kEmpty) {
          // Set the stone color immediately so that the point is no longer
          // empty. We'll patch the list next & prev references when nc is
          // popped off the coord stack.
          points_[nc].bits = color_bits | head;
          coord_stack.push(nc);
        } else if (neighbor_color == other_color) {
          auto nh = chain_head(nc);
          if (neighbor_chains.insert(nh)) {
            points_[nh].liberties_prev -= 1;
          }
        }
      }
    }
  }

  // At the end of the loop, prev is the chain tail: make sure next is
  // invalidated.
  points_[prev].next = Coord::kInvalid;

  // By definition, the uncaptured chain must have only one liberty.
  points_[head].liberties_prev = 1;
  points_[head].bits = color_bits | Point::kIsHeadBit | size;
}

void Position::RebuildChain(Coord c, TaggedPointVisitor* visitor) {
  auto color = point_color(c);
  auto other_color = OtherColor(color);
  auto color_bits = (static_cast<uint16_t>(color) << Point::kColorShift);

  auto head = c;
  auto prev = c;
  int num_liberties = 0;
  int size = 0;

  CoordStack coord_stack;
  coord_stack.push(c);
  visitor->Visit(c, head);

  while (!coord_stack.empty()) {
    c = coord_stack.pop();
    size += 1;

    points_[c].liberties_prev = prev;
    points_[c].bits = color_bits | head;
    points_[prev].next = c;
    prev = c;

    for (auto nc : kNeighborCoords[c]) {
      auto neighbor_color = points_[nc].color();
      if (neighbor_color == other_color || !visitor->Visit(nc, head)) {
        continue;
      }

      if (neighbor_color == Color::kEmpty) {
        num_liberties += 1;
      } else {
        coord_stack.push(nc);
      }
    }
  }

  // At the end of the loop, prev is the chain tail: make sure next is
  // invalidated.
  points_[prev].next = Coord::kInvalid;

  points_[head].liberties_prev = num_liberties;
  points_[head].bits = color_bits | Point::kIsHeadBit | size;
}

Color Position::IsKoish(Coord c) const {
  if (!is_empty(c)) {
    return Color::kEmpty;
  }

  Color ko_color = Color::kEmpty;
  for (Coord nc : kNeighborCoords[c]) {
    auto color = point_color(nc);
    if (color == Color::kEmpty) {
      return Color::kEmpty;
    }
    if (color != ko_color) {
      if (ko_color == Color::kEmpty) {
        ko_color = color;
      } else {
        return Color::kEmpty;
      }
    }
  }
  return ko_color;
}

Position::MoveType Position::ClassifyMove(Coord c) const {
  if (c == Coord::kPass || c == Coord::kResign) {
    return MoveType::kNoCapture;
  }
  if (!is_empty(c) || c == ko_) {
    return MoveType::kIllegal;
  }

  auto result = MoveType::kIllegal;
  auto other_color = OtherColor(to_play_);
  for (auto nc : kNeighborCoords[c]) {
    auto color = point_color(nc);
    if (color == Color::kEmpty) {
      // At least one liberty at nc after playing at c.
      result = MoveType::kNoCapture;
      continue;
    }

    auto num_liberties = num_chain_liberties(nc);
    if (color == other_color) {
      if (num_liberties == 1) {
        // Will capture opponent chain that has a stone at nc.
        return MoveType::kCapture;
      }
    } else if (num_liberties > 1) {
      // Connecting to a same colored chain at nc that has more than one
      // liberty.
      result = MoveType::kNoCapture;
    }
  }
  return result;
}

bool Position::HasNeighboringChain(Coord c, Coord ch) const {
  for (auto nc : kNeighborCoords[c]) {
    if (!is_empty(nc) && chain_head(nc) == ch) {
      return true;
    }
  }
  return false;
}

float Position::CalculateScore(float komi) {
  static_assert(static_cast<int>(Color::kEmpty) == 0, "Color::kEmpty != 0");
  static_assert(static_cast<int>(Color::kBlack) == 1, "Color::kBlack != 1");
  static_assert(static_cast<int>(Color::kWhite) == 2, "Color::kWhite != 2");

  auto territories = CalculatePassAliveRegions();
  for (int i = 0; i < kN * kN; ++i) {
    if (territories[i] == Color::kEmpty) {
      territories[i] = points_[i].color();
    }
  }

  OneTimePointVisitor visitor;

  int score = 0;
  auto score_empty_area = [&visitor, &territories](Coord c) {
    CoordStack coord_stack;
    int num_visited = 0;
    int found_bits = 0;
    for (;;) {
      ++num_visited;
      for (auto nc : kNeighborCoords[c]) {
        auto color = territories[nc];
        if (color == Color::kEmpty && visitor.Visit(nc)) {
          coord_stack.push(nc);
        } else {
          found_bits |= static_cast<int>(color);
        }
      }
      if (coord_stack.empty()) {
        break;
      }
      c = coord_stack.pop();
    }

    if (found_bits == 1) {
      return num_visited;
    } else if (found_bits == 2) {
      return -num_visited;
    } else {
      return 0;
    }
  };

  for (int i = 0; i < kN * kN; ++i) {
    auto color = territories[i];
    if (color == Color::kEmpty) {
      if (visitor.Visit(i)) {
        score += score_empty_area(i);
      }
    } else if (color == Color::kBlack) {
      score += 1;
    } else {
      score -= 1;
    }
  }

  return static_cast<float>(score) - komi;
}

std::array<Color, kN * kN> Position::CalculatePassAliveRegions() const {
  std::array<Color, kN * kN> result;
  for (auto& x : result) {
    x = Color::kEmpty;
  }
  CalculatePassAliveRegionsForColor(Color::kBlack, &result);
  CalculatePassAliveRegionsForColor(Color::kWhite, &result);
  return result;
}

// A _region_ is a connected set of intersections regardless of color.
// A _black-enclosed region_ is a maximal region containing no black stones.
// A black-enclosed region is _small_ if all of its empty intersections are
// liberties of the enclosing black stones.
// A small black-enclosed region is _vital_ to an enclosing chain if all of its
// empty intersections are liberties of that chain. Note that a small
// black-enclosed region may not be vital to any of the enclosing chains. For
// example:
//   . . . . . .
//   . . X X . .
//   . X . . X .
//   . X . . X .
//   . . X X . .
//   . . . . . .
//
// A set of black chains X is _unconditionally alive_ if each chain in X has at
// least two distinct small black-enclosed regions that are vital to it.
// A region enclosed by set of unconitionally alive black chains is an
// unconditionally alive black region.
//
// Given these definitions, Benson's Algorithm finds the set of unconditionally
// alive black regions as follows:
//  - Let X be the set of all black chains.
//  - Let R be the set of small black-enclosed regions of X.
//  - Iterate the following two steps until neither one removes an item:
//    - Remove from X all black chains with fewer than two vital
//      black-enclosed regions in R.
//    - Remove from R all black-enclosed regions with a surrounding stone in a
//      chain not in X.
//
// Unconditionally alive chains are also called pass-alive because they cannot
// be captured by the opponent even if that player always passes on their turn.
// More details:
//   https://senseis.xmp.net/?BensonsDefinitionOfUnconditionalLife
void Position::CalculatePassAliveRegionsForColor(
    Color color, std::array<Color, kN * kN>* result) const {
  // Extra per-chain data required by this implementation of Benson's algorithm.
  struct BensonChain {
    // The number of vital regions that this chain encloses. The BensonChain
    // itself doesn't keep track of which of its neighboring regions are vital,
    // it is sufficient for the BensonChain to track which of its enclosing
    // chains are vital.
    uint16_t num_vital_regions = 0;

    // Whether the chain has been determined to be pass-alive.
    bool is_pass_alive = false;

    std::vector<Coord> enclosed_regions;
  };

  // Extra per-region data required by this implementation of Benson's
  // algorithm.
  struct BensonRegion {
    BensonRegion(int empty_points_begin, int num_empty_points)
        : empty_points_begin(static_cast<uint16_t>(empty_points_begin)),
          num_empty_points(static_cast<uint16_t>(num_empty_points)) {}
    // This region's empty points.
    // See the comments for the regions array below for more details.
    uint16_t empty_points_begin;
    uint16_t num_empty_points;

    // This region's chains.
    // See the comments for the chains array below for more details.
    uint16_t vital_chains_begin;
    uint16_t num_vital_chains = 0;

    std::vector<Coord> vital_chains;

    // A scratch variable that get reused while determining which regions are
    // vital for each chain.
    uint16_t num_liberties_of_chain = 0;

    // A scratch variable that get reused while determining which regions each
    // chain encloses.
    Coord most_recent_chain = Coord::kInvalid;

    // The number of chains that enclose this region.
    // We don't actually need to know which chains they are.
    uint16_t num_enclosing_chains = 0;

    // Whether the region has been determined to be pass-alive.
    bool is_pass_alive = true;
  };

  // Storage for coordinates of empty points in regions.
  // Each BensonRegion has BensonRegion::num_empty_points empty points. The
  // coordinates of the i'th empty point of a region are stored at
  //   empty_points[region->empty_points_begin + i].
  inline_vector<Coord, kN * kN> empty_points;

  // The set of chains for which we're trying to find the pass-alive ones.
  // Indexed by chain_head(c).
  PointMap<BensonChain> chains;

  // The set of regions for which we're trying to find the pass-alive ones.
  PointMap<BensonRegion> regions;

  // If a point c is in an enclosed region (i.e. empty or other_color), then
  // region_indices[c] is the index into the regions array of that region.
  std::array<Coord, kN * kN> region_indices;
  for (auto& x : region_indices) {
    x = Coord::kInvalid;
  }

  // region_chains[region->chains_begin + region->num_enclosing_chains + j]
  inline_vector<Coord, 2 * kN * kN> vital_chains;

  // Used when flood-filling regions.
  CoordStack coord_stack;

  // +-------------------------+
  // | Initialize the regions. |
  // +-------------------------+
  for (int idx = 0; idx < kN * kN; ++idx) {
    Coord region_c(idx);
    if (points_[region_c].color() == color ||
        region_indices[region_c] != Coord::kInvalid) {
      // This point either has a stone of the color we're computing pass-alive
      // territory for, or we've already processed it.
      continue;
    }

    // This is a new region!
    // We will do a few things to initialize it:
    //  - Construct a new BensonRegion at regions[region_c].
    //  - Flood fill region_c into region_indices in all the regions points, so
    //    we can look up the BensonRegion from any point in the region.
    //  - Remember the list of empty points in the region for use later.
    int empty_points_begin = empty_points.size();
    region_indices[region_c] = region_c;

    coord_stack.push(region_c);
    while (!coord_stack.empty()) {
      auto c = coord_stack.pop();

      if (is_empty(c)) {
        empty_points.push_back(c);
      }

      for (auto nc : kNeighborCoords[c]) {
        if (points_[nc].color() != color &&
            region_indices[nc] == Coord::kInvalid) {
          region_indices[nc] = region_c;
          coord_stack.push(nc);
        }
      }
    }
    int num_empty_points = empty_points.size() - empty_points_begin;
    regions.emplace(region_c, empty_points_begin, num_empty_points);
  }

  // +------------------------+
  // | Initialize the chains. |
  // +------------------------+
  TaggedPointVisitor visitor(Coord::kInvalid);
  for (int idx = 0; idx < kN * kN; ++idx) {
    Coord c(idx);
    if (points_[c].color() != color) {
      continue;
    }
    Coord head = chain_head(c);
    if (!visitor.Visit(c, head)) {
      continue;
    }

    // This is a new chain!
    // We do a lot of things to initialize it:
    //  - Construct a BansonChain at the same location as the chain's head, so
    //    it's easy to find.
    //  - For every region that the chain encloses:
    //     - Add the region to the chain's enclosed regions list.
    //     - Increment the region's enclosing chain count.
    //     - Count how many of the region's empty points are liberties for this
    //       chain.
    //    Because we process chains serially (one chain is fully processed
    //    before we move on to the next one), the above per-region steps reuse
    //    some scratch member variables in the BensionRegions for simplicity.
    auto& chain = chains.emplace(head);
    for (auto chain_c = head; chain_c != Coord::kInvalid;
         chain_c = chain_next(chain_c)) {
      visitor.Visit(chain_c, head);
      for (auto nc : kNeighborCoords[chain_c]) {
        auto neighbor_color = points_[nc].color();
        if (neighbor_color == color || !visitor.Visit(nc, head)) {
          continue;
        }

        auto region_idx = region_indices[nc];
        auto& region = regions[region_idx];
        if (region.most_recent_chain != head) {
          // This is the first liberty of this chain that is an empty point of
          // this region. Do some bookkeeping.
          region.most_recent_chain = head;
          region.num_enclosing_chains += 1;
          region.num_liberties_of_chain = 0;
          chain.enclosed_regions.push_back(region_idx);
        }
        if (points_[nc].is_empty()) {
          region.num_liberties_of_chain += 1;
        }
      }
    }

    // Now that we've counted how many of the chain's liberties are empty points
    // of each neighboring region, it's trivial to figure out which of the
    // regions are vital for the chain.
    for (auto region_idx : chain.enclosed_regions) {
      if (regions[region_idx].num_liberties_of_chain ==
          regions[region_idx].num_empty_points) {
        chain.num_vital_regions += 1;
        regions[region_idx].vital_chains.push_back(head);
      }
    }
  }

  // +-------------------------+
  // | Initialization is done. |
  // | Run Benson's algorithm. |
  // +-------------------------+

  // List of chains removed each iteration.
  inline_vector<Coord, kN * kN> removed_chains;

  // The list of candidate pass-alive chains. This is initialized to the full
  // list of chains and pruned.
  auto candidate_chains = chains.coords();
  for (;;) {
    removed_chains.clear();

    // Iterate over remaining chains.
    for (int i = 0; i < candidate_chains.size();) {
      auto chain_idx = candidate_chains[i];
      auto& chain = chains[chain_idx];
      if (chain.num_vital_regions < 2) {
        // This chain has fewer than two vital regions, remove it.
        removed_chains.push_back(chain_idx);
        candidate_chains[i] = candidate_chains.back();
        candidate_chains.pop_back();
      } else {
        i += 1;
      }
    }
    if (removed_chains.empty()) {
      // We didn't remove any chains, we're all done!
      break;
    }

    // For each removed chain, remove every region it's adjacent to.
    for (auto chain_idx : removed_chains) {
      auto& chain = chains[chain_idx];
      for (auto region_idx : chain.enclosed_regions) {
        const auto& r = regions[region_idx];
        for (auto vital_chain_idx : r.vital_chains) {
          chains[vital_chain_idx].num_vital_regions -= 1;
        }
      }
    }
  }

  // candidate_chains now contains only pass-alive chains.
  for (auto chain_idx : candidate_chains) {
    chains[chain_idx].is_pass_alive = true;
  }

  // Now we know which chains are pass-alive, iterate over all the regions,
  // finding which of those are also pass-alive. For a region to be pass-alive,
  // all its enclosing chains must be pass-alive, and all but zero or one empty
  // points must be adjacent to a neighboring chain.

  // A region is only pass-alive if all its enclosing chains are pass-alive.
  for (auto chain_id : chains.coords()) {
    if (!chains[chain_id].is_pass_alive) {
      for (auto& region_id : chains[chain_id].enclosed_regions) {
        regions[region_id].is_pass_alive = false;
      }
    }
  }

  for (auto region_id : regions.coords()) {
    auto& r = regions[region_id];
    if (!r.is_pass_alive) {
      continue;
    }

    // All regions must have at least one empty point, otherwise they'd be dead.
    MG_CHECK(r.num_empty_points != 0);
    if (r.num_enclosing_chains == 0) {
      // Skip regions that have no enclosing chain (the empty board).
      // Because we consider regions that have one empty point that isn't
      // adjacent to an enclosing chain as pass-alive, we don't skip regions
      // that aren't vital to any chains here.
      // TODO(tommadams): num_enclosing_chains is only used here, we can
      // probably get rid of it by doing this check another way.
      continue;
    }

    // A region is only pass-alive if at most one empty point is not adjacent
    // to an enclosing chain.
    int num_interior_points = 0;
    for (uint32_t i = 0; i < r.num_empty_points; ++i) {
      auto c = empty_points[r.empty_points_begin + i];
      bool is_interior = true;
      for (auto nc : kNeighborCoords[c]) {
        if (point_color(nc) == color) {
          is_interior = false;
          break;
        }
      }
      if (is_interior && ++num_interior_points == 2) {
        r.is_pass_alive = false;
        break;
      }
    }
    if (!r.is_pass_alive) {
      continue;
    }

    // This region is pass-alive, mark all the points in the region in the
    // output array.
    // The visitor object has so far only been visited with chain IDs, so we can
    // reuse it without reinitialization to visit regions because chain & region
    // IDs are disjoint.
    auto c = empty_points[r.empty_points_begin];
    coord_stack.push(c);
    visitor.Visit(c, region_id);
    while (!coord_stack.empty()) {
      auto c = coord_stack.pop();
      (*result)[c] = color;
      for (auto nc : kNeighborCoords[c]) {
        if (point_color(nc) != color && visitor.Visit(nc, region_id)) {
          coord_stack.push(nc);
        }
      }
    }
  }
}

bool Position::CalculateWholeBoardPassAlive() const {
  auto territories = CalculatePassAliveRegions();
  for (int i = 0; i < kN * kN; ++i) {
    if (territories[i] == Color::kEmpty && is_empty(i)) {
      return false;
    }
  }
  return true;
}

void Position::UpdateLegalMoves(ZobristHistory* zobrist_history) {
  if (zobrist_history == nullptr) {
    // We're not checking for superko, use the basic result from ClassifyMove to
    // determine whether each move is legal.
    for (int c = 0; c < kN * kN; ++c) {
      if (ClassifyMove(c) == MoveType::kIllegal) {
        points_[c].bits &= ~Point::kIsLegalBit;
      } else {
        points_[c].bits |= Point::kIsLegalBit;
      }
    }
  } else {
    // We're using superko, things are a bit trickier.
    for (int c = 0; c < kN * kN; ++c) {
      switch (ClassifyMove(c)) {
        case Position::MoveType::kIllegal: {
          // The move is trivially not legal.
          points_[c].bits &= ~Point::kIsLegalBit;
          break;
        }

        case Position::MoveType::kNoCapture: {
          // The move will not capture any stones: we can calculate the new
          // position's stone hash directly.
          auto new_hash = stone_hash_ ^ zobrist::MoveHash(c, to_play_);
          if (zobrist_history->HasPositionBeenPlayedBefore(new_hash)) {
            points_[c].bits &= ~Point::kIsLegalBit;
          } else {
            points_[c].bits |= Point::kIsLegalBit;
          }
          break;
        }

        case Position::MoveType::kCapture: {
          // The move will capture some opponent stones: in order to calculate
          // the stone hash, we actually have to play the move.

          Position new_position(*this);
          // It's safe to call AddStoneToBoard instead of PlayMove because:
          //  - we know the move is not kPass.
          //  - the move is legal (modulo superko).
          //  - we only care about new_position's stone_hash and not the rest of
          //    the bookkeeping that PlayMove updates.
          new_position.AddStoneToBoard(c, to_play_);
          auto new_hash = new_position.stone_hash();
          if (zobrist_history->HasPositionBeenPlayedBefore(new_hash)) {
            points_[c].bits &= ~Point::kIsLegalBit;
          } else {
            points_[c].bits |= Point::kIsLegalBit;
          }
          break;
        }
      }
    }
  }
}

void Position::Validate() const {
  // Validate the stone hash.
  std::array<Color, kN * kN> stones;
  for (int i = 0; i < kN * kN; ++i) {
    stones[i] = point_color(i);
  }

  // Validate the chain sizes & liberty counts.
  std::array<bool, kN * kN> validated{};
  for (int i = 0; i < kN * kN; ++i) {
    if (is_empty(i)) {
      MG_CHECK(points_[i].next == Coord::kInvalid);
      continue;
    }
    if (validated[i]) {
      continue;
    }
    auto color = point_color(i);

    std::array<bool, kN * kN> liberties{};
    int size = 0;
    int num_liberties = 0;
    MG_CHECK(!validated[chain_head(i)]);
    for (Coord c = chain_head(i); c != Coord::kInvalid; c = chain_next(c)) {
      size += 1;
      MG_CHECK(point_color(c) == color);
      MG_CHECK(!validated[c]);
      MG_CHECK(!is_legal_move(c));
      if (chain_next(c) != Coord::kInvalid) {
        MG_CHECK(chain_prev(chain_next(c)) == c)
            << "c=" << c << "  " << c << ".prev=" << chain_prev(c) << "  " << c
            << ".next=" << chain_next(c) << "  " << chain_next(c)
            << ".prev=" << chain_prev(chain_next(c));
      }
      for (auto nc : kNeighborCoords[c]) {
        if (is_empty(nc) && !liberties[nc]) {
          liberties[nc] = true;
          num_liberties += 1;
        }
      }
      validated[c] = true;
    }
    MG_CHECK(size == chain_size(i))
        << ToPrettyString() << "\n"
        << Coord(i) << " computed_size:" << size
        << " cached_size:" << chain_size(i) << " chain_head:" << chain_head(i);
    MG_CHECK(num_liberties == num_chain_liberties(i))
        << ToPrettyString() << "\n"
        << Coord(i) << " computed_liberties:" << num_liberties
        << " cached_liberties:" << num_chain_liberties(i);
  }

  MG_CHECK(stone_hash_ == CalculateStoneHash(stones));
}

}  // namespace minigo
