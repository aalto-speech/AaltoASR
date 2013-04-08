#include "Lattice.hh"

Lattice::Lattice()
{
  resize(0, 0, 0);
}

void
Lattice::reset()
{
  for (int f = 0; f < m_frames; f++) {
    if (m_ranges[f].unused())
      break;
    reset_frame(f);
  }
}

void
Lattice::resize(int frames, int positions, int block_size)
{
  m_cells.resize(frames);
  m_ranges.resize(frames);
  m_blocks.resize(frames);

  m_block_size = block_size;

  for (int i = 0; i < frames; i++) {
    m_cells[i].resize(m_block_size);
    m_blocks[i].size = m_cells[i].size();
  }

  m_frames = frames;
  m_positions = positions;
}

// NOTES:
// - all frames after an unused frame must be unused
// - cells in unused frames should have cell.from = -1
// - removes the paths which do not go through the given (frame,position)
void
Lattice::move(int frame, int position)
{
  assert(position >= 0); // REMOVE
  assert(frame > 0); // REMOVE
  
  // Clear ranges before frame
  for (int f = 0; f < frame; f++) {
    Range &range = m_ranges[f];
    
    // All frames after an unused range are unused
    if (range.unused())
      return;
    
    reset_frame(f);
  }
  
  // Move active part of the lattice and ranges
  for (int source_f = frame; source_f < m_frames; source_f++) {

    int target_f = source_f - frame;
    Range &source_range = m_ranges[source_f];
    
    // All ranges after an unused range are unused
    if (source_range.unused()) {
      return;
    }
    
    // Lets swap the frames
    swap(source_f, target_f);
    
    Range &target_range = m_ranges[target_f];
    Block &target_block = m_blocks[target_f];
    
    // Clear the bottom source part which falls out of border
    if (target_range.start < position)
      reset_frame(target_f, position, target_range.end);
    
    // Update from -pointers
    // At left we keep from -pointers -1
    // and at bottom 0    
    for (int i = target_range.start; i < target_range.end; i++) {
      at(target_f,i).from -= position;
      if (target_f == 0) at(target_f,i).from = -1;
      else if (i - position == 0) at(target_f,i).from = 0;
    }
    
    // Update ranges
    // FIXME: this way block.min can be below zero..is this a problem?
    target_range.start -= position;
    target_range.end -= position;
    target_block.min -= position;
    target_block.max -=position;
  }
}

void
Lattice::check_active()
{
  // Checks if active area is consistent
  for (int f = 0; f < m_frames; f++) {
    Range &range = m_ranges[f];
    if (range.unused())
      continue;

    for (int p = range.start; p < range.end; p++) {
      Cell &cell = at(f, p);

      // Check pointer
      if (f == 0) {
	if (cell.from >= 0)
	  assert(false);
      }
      else {
	if (cell.from < 0)
	  assert(false);

	Range &prev_range = m_ranges[f - 1];
	if (cell.from < prev_range.start || cell.from >= prev_range.end)
	  assert(false);
      }
    }
  }
}

void
Lattice::check_empty(int frame)
{
  // Checks if cells after (and including) frame are cleared
  // properly.
  for (int f = frame; f < m_frames; f++) {
    if (range(f).start >= 0 || range(f).end >= 0)
      assert(false);
  }
}


/// On Implementation: no vector is ever resized;
/// a new one is instead created in order to keep
/// each frame completely in one memory page (?)
/// resize-method was tested and seemed to be much slower
void
Lattice::add_block(int frame)
{
  assert(frame >= 0 && frame < m_frames);

  Block &block = m_blocks[frame];
  
  // Initialize new vector
  int old_size = block.size;
  int old_min = block.min;
  int old_max = block.max;
  int new_size = old_size + m_block_size;
  std::vector<Cell> new_cells;
  new_cells.resize(new_size);

  // Raise the upper limit
  block.max += m_block_size;

  // Copy old contents to the new vector
  if (old_min != -1)
    for (int i = old_min; i < old_max; i++)
      new_cells[i - block.min] = at(frame, i);
  
  // Swap old <-> new
  m_cells[frame].swap(new_cells);
  block.size += m_block_size;
}

Lattice::Block&
Lattice::block(int frame)
{
  return m_blocks[frame];
}

void
Lattice::frame_stats(int frame)
{ 
  std::cout << std::endl
	    << "statistics for frame " << frame << std::endl
	    << "range: " << m_ranges[frame].start << " - " << m_ranges[frame].end
	    << ", block: " << m_blocks[frame].min << " - " << m_blocks[frame].max
	    << ", size: " << m_blocks[frame].size
	    << ", actual: " << m_cells[frame].size() << std::endl;
}

void
Lattice::stats() 
{
  std::cout << std::endl
	    << "statistics for the whole lattice" << std::endl
	    << "m_frames = " << m_frames << " actual = " << m_cells.size() << std::endl
	    << "m_positions = " << m_positions << std::endl
	    << "m_block_size = " << m_block_size << std::endl;

}

void
Lattice::swap(int frame_1, int frame_2)
{
  Range temp_range = m_ranges[frame_1];
  Block temp_block = m_blocks[frame_1];

  m_cells[frame_1].swap(m_cells[frame_2]);

  m_ranges[frame_1] = m_ranges[frame_2];
  m_blocks[frame_1] = m_blocks[frame_2];
  m_ranges[frame_2] = temp_range;
  m_blocks[frame_2] = temp_block;
}
