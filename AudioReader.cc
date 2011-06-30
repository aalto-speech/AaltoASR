#include <limits.h>
#include "AudioReader.hh"
#include <string>
#include <iostream>
#include <fcntl.h>

namespace aku {

AudioReader::AudioReader()
  : m_sndfile(NULL),
    m_little_endian(true),
    m_raw(false),
    m_eof_sample(INT_MAX),
    m_buffer_size(0),
    m_start_sample(-INT_MAX),
    m_end_sample(-INT_MAX),
    m_file_sample(0)
{
  sf_info.format = 0;
}

AudioReader::AudioReader(int buffer_size)
  : m_sndfile(NULL),
    m_little_endian(true),
    m_eof_sample(INT_MAX),
    m_buffer_size(0),
    m_start_sample(-INT_MAX),
    m_end_sample(-INT_MAX),
    m_file_sample(0)
{
  sf_info.format = 0;
}

AudioReader::~AudioReader()
{
  close();
}

void // private
AudioReader::reset()
{
  m_eof_sample = INT_MAX;
  m_start_sample = -INT_MAX;
  m_end_sample = -INT_MAX;
  m_file_sample = 0;
  sf_info.format = 0;
}

void // private
AudioReader::resize(int start, int end)
{
  assert(end > start);
  int offset = start - m_start_sample;
  int new_len = end - start;
  int old_len = m_end_sample - m_start_sample;
  if (new_len > m_buffer_size) {
    m_buffer.resize(new_len);
    m_buffer_size = new_len;
  }
  m_start_sample = start;
  m_end_sample = end;

  if (offset == 0)
    return;

  int copy_start = std::max(0, offset);
  int copy_end = std::min(old_len, new_len + offset);

  if (offset < 0)
  {
    for (int i = copy_end - 1; i >= copy_start; i--)
      m_buffer[i - offset] = m_buffer[i];
  }
  else
  {
    for (int i = copy_start; i < copy_end; i++)
      m_buffer[i - offset] = m_buffer[i];
  }
}

void
AudioReader::open(const char *filename, int sample_rate)
{
  close();
  reset();

  if (!m_raw)
    m_sndfile = sf_open(filename, SFM_READ, &sf_info);
  
  if (m_sndfile == NULL)
  {
    // Try opening in RAW mode
    sf_info.format = SF_FORMAT_RAW | SF_FORMAT_PCM_16;
    if (m_little_endian)
      sf_info.format |= SF_ENDIAN_LITTLE;
    else
      sf_info.format |= SF_ENDIAN_BIG;
    sf_info.samplerate = sample_rate;
    sf_info.channels = 1;
    m_sndfile = sf_open(filename, SFM_READ, &sf_info);
    if (m_sndfile == NULL)
      throw std::string("AudioReader::open(): could not open file:") +
        std::string(filename);
  }
  check_audio_parameters();
}

void
AudioReader::open(FILE *file, int sample_rate, bool shall_close_file, bool stream)
{
  close();
  reset();

  if (!stream && !m_raw)
    m_sndfile = sf_open_fd(fileno(file), SFM_READ, &sf_info, shall_close_file);
  
  if (m_sndfile == NULL)
  {
    if (stream)
      sf_info.seekable = 0;
    else {
      sf_info.seekable = 1;
      rewind(file); // FIXME: How about non-seekable streams?
    }
    // Try opening in RAW mode
    sf_info.format = SF_FORMAT_RAW | SF_FORMAT_PCM_16;
    if (m_little_endian)
      sf_info.format |= SF_ENDIAN_LITTLE;
    else
      sf_info.format |= SF_ENDIAN_BIG;
    sf_info.samplerate = sample_rate;
    sf_info.channels = 1;
    m_sndfile = sf_open_fd(fileno(file), SFM_READ, &sf_info, shall_close_file);
    if (m_sndfile == NULL)
      throw std::string("AudioReader::open(): sf_open_fd() failed");
  }

  check_audio_parameters();
}

void // private
AudioReader::check_audio_parameters()
{
  if (sf_info.channels != 1)
    throw std::string("AudioReader: sorry, audio files with multiple "
		      "channels not supported");

  if ((sf_info.format & SF_FORMAT_SUBMASK) != SF_FORMAT_PCM_16)
    fprintf(stderr, "WARNING: AudioReader::check_audio_parameters(): "
	    "sample format not SF_FORMAT_PCM_16 but 0x%08x\n",
	    sf_info.format);
}

void
AudioReader::close()
{
  if (m_sndfile == NULL)
    return;

  if (sf_close(m_sndfile) != 0)
    throw std::string("AudioReader::close(): sf_close() failed\n");
  m_sndfile = NULL;
  sf_info.format = 0;
}

void // private
AudioReader::read_from_file(int start, int end)
{
  if (start != m_file_sample) {
    if (sf_info.seekable)
      seek(start);
    else
      throw std::string("Trying to seek a non-seekable FILE\n");
  }

  int index = start - m_start_sample;
  int samples_to_read = end - start;

  // Fill zeros before file start
  //
  while (start < 0 && samples_to_read > 0) {
    samples_to_read--;
    m_buffer[index++] = 0;
    start++;
  }

  // Read samples from the file
  //
  if (start < m_eof_sample)
    while (samples_to_read > 0) {

      sf_count_t samples_read = 
	sf_read_short(m_sndfile, &m_buffer[index], samples_to_read);
      assert(samples_read >= 0);

      if (samples_read == 0) {
	m_eof_sample = m_file_sample;
	break;
      }
      m_file_sample += samples_read;
      samples_to_read -= samples_read;
      index += samples_read;
    }    

  // Fill zeros after file end
  //
  while (samples_to_read-- > 0)
    m_buffer[index++] = 0;
}

// INVARIANTS:
//
// - outside this function, sample 'start' is located at buffer
// position 0
void
AudioReader::fetch(int start, int end)
{
  int old_start_sample = m_start_sample;
  int old_end_sample = m_end_sample;
  resize(start, end);

  if (start < old_start_sample)
    read_from_file(start, std::min(end, old_start_sample));
  if (end > old_end_sample)
    read_from_file(std::max(start, old_end_sample), end);
}

void // private
AudioReader::seek(int sample)
{
  if (sample < 0)
    sample = 0;
  if (sample == m_file_sample)
    return;
  if (sample >= m_eof_sample)
    return;
  if (sf_info.seekable == 0)
    throw std::string("AudioReader::seek(): non-seekable file");
  sf_count_t ret = sf_seek(m_sndfile, sample, SEEK_SET);
  if (ret < 0)
    throw std::string("AudioReader::seek(): sf_seek() failed");
  if (ret != sample)
    throw std::string("AudioReader::seek(): seek() seeked to wrong place?!");
  m_file_sample = sample;
}

}
