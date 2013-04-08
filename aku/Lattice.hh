#ifndef LATTICE_HH
#define LATTICE_HH

#include <vector>
#include <assert.h>
#include <iostream>


#define INF (float)1e24

/// A 2-dimensional lattice for the Viterbi search.
/** 
 * The lattice maintains a 2-dimensional lattice which can be used for
 * computing best paths through an observation sequence.  The class
 * maintains information about the active part of the lattice, so that
 * pruning can be applied. 
 *
 * IMPLEMENTATION NOTES
 * -A vector of cells is reserved for each frame separately
 * -The size of these vectors is raised in fixed blocks, and never decreased,
 *  so after couple of first windows the memory consumption should stabilize
 * -Only active cells are preserved in memory
 * -Block.min <= Range.start <= Range.end <= Block.max
 **/
class Lattice {
public:

  /// Lattice cell which stores information about probabilities and best paths.
  struct Cell {
    /// A structure
    inline Cell() : from(-1), transition_index(-1), log_prob(-INF) { }
    inline void clear() { from = -1; transition_index = -1; log_prob = -INF; }
    inline bool unused() { return (from < 0 && log_prob == -INF); }

    /// The best position leading to this position on the previous frame.
    int from;			

    /// The index of the best transition to this position.
    int transition_index;	

    /// The log-probability of the best path ending in this position and frame.
    float log_prob;
  };

  /// A structure for storing the active range of states for a frame.
  struct Range {
    inline Range() : start(-1), end(-1) { }
    inline void clear() { start = -1; end = -1; }
    inline bool unused() { return start < 0; }
    inline bool has(int p) { return (p >= start && p < end); }

    /// The first active position on this frame.
    int start;

    /// The first inactive position on this frame.
    int end;
  };

  /// A structure for storing the info about the memory reserved for a frame
  struct Block {
    inline Block() : min(-1), max(-1) { }
    inline void clear() { min = -1; max = -1; }
    inline bool has(int p) { return (p >= min && p < max); }
    /// The full size reserved for the frame
    int size;
    /// The first possible position
    int min;
    /// The first impossible position
    int max;
  };

  Lattice();

  /// Reset the whole lattice.
  void reset();

  /// Resize the lattice.  FIXME: does this lose the contents.
  void resize(int frames, int positions, int block_size);

  /// Reposition the lattice.
  /**
   * All cells (f,p) for which (f >= frame) and (p >= position) are
   * moved to (f-frame, p-position), and the source cells are cleared.
   * Ranges and from-pointers are also updated accordingly. */
  void move(int frame, int position);

  /// Return the active range for the given frame.
  inline Range &range(int frame);
  
  /// Reset a frame of the lattice so that start..end is left active.
  /// Handles memory reservations automatically
  inline void reset_frame(int frame, int start = -1, int end = -1);

  /// Return the requested lattice cell.
  inline Cell &at(int frame, int position);

  /// Return the number of frames in the lattice.
  inline int frames() { return m_frames; }

  /// Return the number of positions in the lattice.
  inline int positions() { return m_positions; }

  // Debugging
  void check_active();
  void check_empty(int frame);
  void stats();
  void frame_stats(int frame);

private:
  /// Cells of the lattice.
  std::vector<std::vector<Cell> > m_cells;
  
  /// The "width" of the lattice.
  int m_frames;			
  
  /// The "height" of the lattice.
  int m_positions;		
  
  /// Active ranges for each frame.
  std::vector<Range> m_ranges;	
  
  /// Memory positions for each frame.
  std::vector<Block> m_blocks;
  
  /// For how many cells is memory reserved each time
  int m_block_size;
  
  /// Reserves a new memory block to the 'back' of the vector
  void add_block(int frame);

  /// Swaps the contents of two frames, used by move()
  void swap(int frame_1, int frame_2);

  /// Memory limits for the given frame.
  Block &block(int frame);
};

Lattice::Range&
Lattice::range(int frame)
{
  assert(frame >= 0 && frame < m_frames);
  return m_ranges[frame];
}

/// Assumes that range is set before fetching cells (Viterbi!)
Lattice::Cell&
Lattice::at(int frame, int position)
{
  Block &block = m_blocks[frame];
  Range &range = m_ranges[frame];

  assert(range.has(position));
  assert(block.has(position));

  return m_cells[frame][position - block.min];
}

/// Reset_frame has some assumptions: If Viterbi-search is modified, 
/// this will probably be the first one to cause problems
/// Reset_frame can be used to:
/// -Clear frame (new_start == -1, new_end == -1)
/// -Initialize frame (range.start == -1, range.end == -1)
/// -Expand frame end
/// Each frame must be cleared every second window
void
Lattice::reset_frame(int frame, int new_start, int new_end)
{
  Range &range = m_ranges[frame];
  Block &block = m_blocks[frame];

  // Clear the frame & block
  if (new_start == -1 && new_end == -1) {
    if (range.start != -1 && range.end != -1) {
      for (int i = range.start; i < range.end; i++)
	at(frame, i).clear();
    }
    block.clear();
  }

  // If frame was empty, lets do some initialization
  else if (range.start == -1 && range.end == -1) {
    if (new_start != -1) {
      while (block.size < new_end - new_start)
	add_block(frame);
      block.min = new_start;
      block.max = new_start + block.size;
    }
  }

  // Frame is to be expanded
  else {
    // now we have situation where range=(-1,XX) and block not set at all
    // this is needed to get things in sync with Viterbi
    if (range.start == -1 && new_start != -1) {
      block.min = new_start;
      block.max = new_start + block.size;
    }
    
    // add new block of memory
    while (new_end > block.max)
      add_block(frame);

    // clear cells that are left outside range
    if (new_start > range.start && range.start != -1)
      for (int i = range.start; i < new_start; i++)
	at(frame, i).clear();
    if (new_end < range.end)
      for (int i = range.end - 1; i >= new_end; i--)
	at(frame, i).clear();
  }
  
  range.start = new_start;
  range.end = new_end;
}

#endif /* LATTICE_HH */

