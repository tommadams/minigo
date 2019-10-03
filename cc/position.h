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

#ifndef CC_POSITION_H_
#define CC_POSITION_H_

#include <array>
#include <cstdint>
#include <memory>
#include <string>

#include "cc/color.h"
#include "cc/constants.h"
#include "cc/coord.h"
#include "cc/inline_vector.h"
#include "cc/logging.h"
#include "cc/zobrist.h"

namespace minigo {
extern const std::array<inline_vector<Coord, 4>, kN * kN> kNeighborCoords;

// Position represents a single board position.
// It tracks the stones on the board and their chains, and contains the logic
// for removing chains with no remaining liberties and merging neighboring
// chains of the same color.
//
// Since the MCTS code makes a copy of the board position for each expanded
// node in the tree, we aim to keep the data structures as compact as possible.
// This is in tension with our other aim of avoiding heap allocations where
// possible, which means we have to preallocate some pools of memory.
class Position {
 public:
  // A default constructed Point represents an empty point that's legal to
  // play.
  struct Point {
    static constexpr int kIsHeadBit = 0x8000;
    static constexpr int kIsLegalBit = 0x4000;
    static constexpr int kColorBits = 0x3000;
    static constexpr int kColorShift = 12;
    static constexpr int kSizeBits = 0x01ff;

    // Returns the color of the chain at c, or Color::kEmpty.
    Color color() const {
      return static_cast<Color>((bits & Point::kColorBits) >>
                                Point::kColorShift);
    }

    // Returns true if the point is empty.
    bool is_empty() const { return (bits & Point::kColorBits) == 0; }

    // Returns true if the point is a legal move.
    bool is_legal_move() const { return (bits & Point::kIsLegalBit) != 0; }

    // If this point is the head of a chain: holds the number of liberties of
    // the chain. If this point is empty: holds 0. Otherwise: the previous point
    // in the chain.
    uint16_t liberties_prev = 0;

    // The next point in a chain, or Coord::kInvalid.
    uint16_t next = Coord::kInvalid;

    //  f e d c b a 9 8 7 6 5 4 3 2 1 0
    // +-+-+---+-----+-----------------+
    // |H|L|C C|X X X|S S S S S S S S S|
    // +-+-+---+-----+-----------------+
    // Packed bit field:
    //  H : If 1, this point is the head of a chain or empty region.
    //  L : If 1, this point is a legal move.
    //  C : This point's color: empty, black, white.
    //  X : Reserved for future expansion.
    //  S : If this point is the head of a chain or empty region: the size of
    //  that chain or empty region.
    //      Otherwise: the index of the head of the chain or empty region.
    uint16_t bits = kIsLegalBit;
  };

  using Points = std::array<Point, kN * kN>;

  // State required to undo a call to PlayMove.
  struct UndoState {
    UndoState(Coord c, Color to_play, Coord ko)
        : c(c), to_play(to_play), ko(ko) {}
    Coord c;
    Color to_play;
    Coord ko;
    inline_vector<Coord, 4> captures;
  };

  // Calculates the Zobrist hash for an array of stones. Prefer using
  // Position::stone_hash() if possible.
  static zobrist::Hash CalculateStoneHash(
      const std::array<Color, kN * kN>& stones);

  // Interface used to enforce positional superko based on the Zobrist hash of
  // a position.
  class ZobristHistory {
   public:
    virtual bool HasPositionBeenPlayedBefore(
        zobrist::Hash stone_hash) const = 0;
  };

  explicit Position(Color to_play);
  Position(const Position&) = default;
  Position& operator=(const Position&) = default;

  // Plays the given move and updates which moves are legal.
  // If zobrist_history is non-null, move legality considers positional superko.
  // If zobrist_history is null, positional superko is not considered when
  // updating the legal moves, only ko.
  // Returns an UndoState object that allows the move to be undone.
  UndoState PlayMove(Coord c, Color color = Color::kEmpty,
                     ZobristHistory* zobrist_history = nullptr);

  // Undoes the most recent call to PlayMove.
  // zobrist_history should not include the move to be undone.
  void UndoMove(const UndoState& undo,
                ZobristHistory* zobrist_history = nullptr);

  // TODO(tommadams): Do we really need to store this on the position? Return
  // the number of captured stones from AddStoneToBoard and track the number of
  // captures in the player.
  const std::array<int, 2>& num_captures() const { return num_captures_; }

  // Calculates the score from B perspective. If W is winning, score is
  // negative.
  float CalculateScore(float komi);

  // Calculates all pass-alive region that are enclosed by chains of `color`
  // stones.
  // Elements in the returned array are set to `Color::kBlack` or
  // `Color::kWhite` if they belong to a pass-alive region or `Color::kEmpty`
  // otherwise. Only intersections inside the enclosed region are set,
  // intersections that are part of an enclosing chain are set to
  // `Color::kEmpty`. Concretely, given the following position:
  //   X . X . O X .
  //   X X X X X X .
  //   . . . . . . .
  // The returned array will be set to:
  //   . X . X X . .
  //   . . . . . . .
  std::array<Color, kN * kN> CalculatePassAliveRegions() const;

  // Returns true if the whole board is pass-alive.
  bool CalculateWholeBoardPassAlive() const;

  // Returns true if playing this move is legal.
  // Does not check positional superko.
  // legal_move(c) can be used to check for positional superko.
  enum class MoveType {
    // The position is illegal:
    //  - a stone is already at that position.
    //  - the move is ko.
    //  - the move is suicidal.
    kIllegal,

    // The move will not capture an opponent's chain.
    // The move is not necessarily legal because of superko.
    kNoCapture,

    // The move will capture an opponent's chain.
    // The move is not necessarily legal because of superko.
    kCapture,
  };
  MoveType ClassifyMove(Coord c) const;

  std::string ToSimpleString() const;
  std::string ToPrettyString(bool use_ansi_colors = true) const;

  Color to_play() const { return to_play_; }
  int n() const { return n_; }
  Coord ko() const { return ko_; }
  zobrist::Hash stone_hash() const { return stone_hash_; }
  const Points& points() const { return points_; }

  // Returns the color of the chain at c, or Color::kEmpty.
  Color point_color(Coord c) const { return points_[c].color(); }

  // Return true if the point at c is empty.
  bool is_empty(Coord c) const { return points_[c].is_empty(); }

  // Returns true if the point at c is a legal move.
  bool is_legal_move(Coord c) const {
    MG_DCHECK(c <= Coord::kResign);
    return c == Coord::kPass || c == Coord::kResign ||
           points_[c].is_legal_move();
  }

  // Returns the number of liberties of the chain at c.
  // Returns 0 if the point at c is empty.
  int num_chain_liberties(Coord c) const {
    MG_DCHECK(!is_empty(c));
    return static_cast<int>(points_[chain_head(c)].liberties_prev);
  }

  // Returns the head of the chain at c.
  // All stones in a chain have the same head.
  Coord chain_head(Coord c) const {
    MG_DCHECK(!is_empty(c));
    return is_head(c) ? c
                      : static_cast<Coord>(points_[c].bits & Point::kSizeBits);
  }

  // Returns the previous stone in the chain at c o/ Coord::kInvalid if this is
  // the head of the list.
  Coord chain_prev(Coord c) const {
    MG_DCHECK(!is_empty(c));
    return is_head(c) ? Coord::kInvalid : points_[c].liberties_prev;
  }

  // Returns the next stone in the chain c or Coord::kInvalid if this is the
  // tail of the list.
  Coord chain_next(Coord c) const {
    MG_DCHECK(!is_empty(c));
    return points_[c].next;
  }

  // Returns the size of chain region at c.
  int chain_size(Coord c) const {
    MG_DCHECK(!is_empty(c));
    return static_cast<int>(points_[chain_head(c)].bits & Point::kSizeBits);
  }

  // Validates the internal consitency of the Position.
  // This should only be called in tests & debug builds.
  void Validate() const;

  // The following methods are protected to enable direct testing by unit tests.
 protected:
  // Sets the pass alive regions for the given color in result.
  // The caller is responsible for initializing all elements in `result` to
  // `Color::kEmpty` before calling.
  void CalculatePassAliveRegionsForColor(
      Color color, std::array<Color, kN * kN>* result) const;

  // Returns color C if the position at idx is empty and surrounded on all
  // sides by stones of color C.
  // Returns Color::kEmpty otherwise.
  Color IsKoish(Coord c) const;

  // Adds the stone to the board.
  // Removes newly surrounded opponent chains.
  // DOES NOT update legal_moves_: callers of AddStoneToBoard must explicitly
  // call UpdateLegalMoves afterwards (this is because UpdateLegalMoves uses
  // AddStoneToBoard internally).
  // Updates liberty counts of remaining chains.
  // Updates num_captures_.
  // If the move captures a single stone, sets ko_ to the coordinate of that
  // stone. Sets ko_ to kInvalid otherwise.
  // Returns a list of the neighbors of c that belonged to chains that were
  // captured by this move.
  inline_vector<Coord, 4> AddStoneToBoard(Coord c, Color color);

  // Updates legal_moves_.
  // If zobrist_history is non-null, this takes into account positional superko.
  void UpdateLegalMoves(ZobristHistory* zobrist_history);

 private:
  // Simple array that marks when points on a board have been visited.
  // Only supports visiting each point once.
  class OneTimePointVisitor {
   public:
    bool Visit(Coord c) {
      if (!visited_[c]) {
        visited_[c] = true;
        return true;
      }
      return false;
    }

   private:
    std::array<bool, kN * kN> visited_{};
  };

  // Array that marks when points on a board have been visited.
  // Uses 2x the storage as OneTimePointVisitor but allows points to be visited
  // multiple times, as long as the visit tags are different each time.
  class TaggedPointVisitor {
   public:
    explicit TaggedPointVisitor(uint16_t invalid_tag)
        : invalid_tag_(invalid_tag) {
      for (auto& x : visited_) {
        x = invalid_tag;
      }
    }

    bool HasAnyVisit(Coord c) const { return visited_[c] != invalid_tag_; }

    bool Visit(Coord c, uint16_t tag) {
      if (visited_[c] != tag) {
        visited_[c] = tag;
        return true;
      }
      return false;
    }

   private:
    std::array<uint16_t, kN * kN> visited_;
    const uint16_t invalid_tag_;
  };

  // Removes the chain with a stone at the given coordinate from the board,
  // updating the liberty counts of neighboring chains.
  void RemoveChain(Coord c);

  // Called as part of UndoMove for the given color at point capture_c.
  // Replaces the previously captured stones at point chain_c.
  void UncaptureChain(Color color, Coord capture_c, Coord chain_c);

  // Called as part of UndoMove.
  // Rebuilds the chain at c. The chain's head changes to c.
  void RebuildChain(Coord c, TaggedPointVisitor* visitor);

  // Returns true if the point at coordinate c neighbors a chain with head ch.
  bool HasNeighboringChain(Coord c, Coord ch) const;

  uint16_t is_head(Coord c) const {
    return points_[c].bits & Point::kIsHeadBit;
  }

  Points points_;

  Color to_play_;
  Coord ko_ = Coord::kInvalid;

  // Number of captures for (B, W).
  std::array<int, 2> num_captures_{{0, 0}};

  int n_ = 0;

  // Zobrist hash of the stones. It can be used for positional superko.
  // This has does not include number of consecutive passes or ko, so should not
  // be used for caching inferences.
  zobrist::Hash stone_hash_ = 0;
};

}  // namespace minigo

#endif  // CC_POSITION_H_
