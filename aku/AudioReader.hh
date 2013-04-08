#ifndef AUDIOREADER_HH
#define AUDIOREADER_HH

#include <sndfile.h>
#include <vector>
#include <assert.h>

namespace aku {

/** A class for reading and buffering audio streams. 
 *
 * Currently, the class expects only streams with one channel.  For
 * the purposes of the speech recognition system in CIS, warnings are
 * issued if the stream differs from: 1 channels, 16 bits,
 * AF_SAMPFMT_TWOSCOMP.
 *
 * \bugs read_to() should use automatic pointers to avoid memory
 * leaks.  Uses ints for indexing samples, so the maximum size of the
 * audio stream is around 2G samples.
 */
class AudioReader {
public:

  /** Create a reader with the default buffer size (4096 samples). */
  AudioReader();

  /** Create a reader with a given buffer size. */
  AudioReader(int buffer_size);
  
  /** Destroy the reader (closes file depending on \ref
   * m_shall_close_file).
   */
  ~AudioReader();

  /** Open a named file.
   * \param file = the file to read the audio stream from
   * \param sample_rate = sample rate (needed if the file is RAW)
   */
  void open(const char *filename, int sample_rate);

  /** Initialize reading from a file handle. 
   *
   * \param file = the file to read the audio stream from
   * \param sample_rate = sample rate (needed if the file is RAW)
   * \param shall_close_file = will the class take care of closing the file?
   * \param stream = set this if we are streaming
   */
  void open(FILE *file,
            int sample_rate,
            bool shall_close_file = false,
            bool stream = false);

  /** Close the file, but only if \ref m_shall_close_file is \c true. */
  void close();

  /** Read samples from the file to buffer.  It is allowed to fetch
   * samples outside the file.  Zero samples are generated in that
   * case.  However, jumping directly over end of file generates a
   * fatal error.
   *
   * \param start = the first frame to read into the buffer
   * \param end = the first sample to NOT read into the buffer
   */
  void fetch(int start, int end);

  /** The index of the sample corresponding to the end of file
   * (INT_MAX if not set yet). */
  int eof_sample() const { return m_eof_sample; }
	
  /** Access the audio data in the range requested with \ref fetch().
   * Error is generated if sample is requested outside the range of
   * the last \fetch() call.  Note that samples outside the physical
   * file can be generated by fetch().
   */
  short operator[](int sample) const
  { 
    assert(sample >= m_start_sample && sample < m_end_sample);
    return m_buffer[sample - m_start_sample];
  }

  /** Return the sample rate of the audio file. */
  int sample_rate() const { return sf_info.samplerate; }

  /** Return the number of samples in the file. */
  int num_samples() const { return sf_info.frames; }

  /** Sets endianess. */
  void set_little_endian(bool endian) { m_little_endian = endian; }

  /** Force to RAW mode */
  void enforce_raw(bool raw) { m_raw = raw; }

  /** The sample format. */
  SF_INFO sf_info;

protected:

  /** Set the size and position of the audio buffer while preserving
   * the old content between the requested range.  Does nothing, if
   * the position and size is same as before.  After the call, the
   * sample 'start' is located at the beginning of the buffer. */
  void resize(int start, int pos);

  /** Reset the state for opening a new stream. */
  void reset();
  
  /** Check if the parameters are not standard. */
  void check_audio_parameters();

  /** Read samples from the file.  Sets \ref eof_sample if EOF
   * reached. */
  void read_from_file(int start, int end);

  /** Seek to given sample.  Does not read anything; just moves the
   * file pointer for the next read.  The buffer contents will be
   * undefined after the call.
   */
  void seek(int sample);

  /** Handle to the audio file. */
  SNDFILE *m_sndfile;

  /** Close the file eventually */
  bool m_shall_close_file;

  /** Endianess for RAW-files */
  bool m_little_endian;

  /** File mode enforced to RAW? */
  bool m_raw;

  /** The index of the sample corresponding to end of file (INT_MAX
   * if not set yet). */
  int m_eof_sample;

  /** The size of the ring buffer. */
  int m_buffer_size;

  /** The sample number of the first valid sample. */
  int m_start_sample;

  /** The sample number of the last valid sample PLUS ONE. */
  int m_end_sample;

  /** The sample number the file points to. */
  int m_file_sample;

  /** The buffer containing the audio data. */
  std::vector<short> m_buffer;

};

}

#endif /* AUDIOREADER_HH */