#ifndef STATEPROBREADER_HH
#define STATEPROBREADER_HH

#include "Acoustics.hh"

/** Class for reading state probability stream (SPS)
 *
 * The class reads probability stream from a given FILE pointer, and
 * stores a fixed amount of past frames (default: only the current
 * frame) in the internal buffer.  The past frames which fall out of
 * the internal buffer are irreversibly lost.  The user can specify
 * how many frames are preserved.
 *
 * IMPLEMENTATION NOTES:
 * - frame f goes to ((f % m_frames)*m_num_states) in the internal buffer
 * - Currently we do not use seek at all.  FIXME: Do we need it?
 *
 * INVARIANTS:
 * - size of the buffer must be a multiple of num_models
 **/
class StateProbReader : public Acoustics {
public:
  StateProbReader();

  /// Open a new stream, and read the header information.
  void open(FILE *file, int frames = 1);

  // Methods inherited from Acoustics
  virtual ~StateProbReader();
  virtual bool go_to(int frame);

private:
  /// Returns the pointer to the given frame in the internal buffer.
  float *frame_ptr(int frame);

  FILE *m_file;

  /// Index of the last frames in the buffer.
  int m_last_frame;

  /// The size of the buffer in frames.
  int m_frames;

  /// Internal read buffer.
  std::vector<float> m_read_buffer;
};

#endif /* STATEPROBREADER_HH */
