#include <assert.h>
#include "AudioPlayer.hh"
#include <string>

static void 
audio_callback(void *userdata, Uint8 *stream, int stream_length)
{
  AudioPlayer::CallbackData *callback_data = 
    (AudioPlayer::CallbackData*)userdata;
  std::vector<short> &buffer = callback_data->buffer;

  // Fill the audio output buffer
  int stream_pos = 0;
  int samples_to_write = stream_length / sizeof(short);
  int bytes_to_write = stream_length;
  int samples_available = buffer.size() - callback_data->pos;

  if (samples_available < samples_to_write) {
    samples_to_write = samples_available;
    bytes_to_write = samples_to_write * sizeof(short);
  }

  for (int i = 0; i < samples_to_write; i++) {
    int index = callback_data->pos + i;
    ((short*)stream)[i] = buffer.at(index);
  }
  callback_data->pos += samples_to_write;
  stream_pos += bytes_to_write;

  // Fill rest with zeros
  if (stream_length > stream_pos)
    memset(&stream[stream_pos], 0, stream_length - stream_pos);
}

AudioPlayer::AudioPlayer(int freq) 
{
  // Set the audio format
  SDL_AudioSpec wanted_spec;
  wanted_spec.freq = freq;
  wanted_spec.format = AUDIO_S16SYS;
  wanted_spec.channels = 1;
  wanted_spec.samples = 4096;
  wanted_spec.callback = audio_callback;
  wanted_spec.userdata = &m_callback_data;
  m_callback_data.pos = 0;

  if (SDL_OpenAudio(&wanted_spec, &m_audio_out_spec) < 0) {
    fprintf(stderr, "couldn't open audio: %s\n", SDL_GetError());
    exit(1);
  }
}

AudioPlayer::~AudioPlayer() 
{
  SDL_PauseAudio(1);
  SDL_CloseAudio();
}

void 
AudioPlayer::enqueue(short *buf, int length) 
{
  SDL_LockAudio();
  // Clean buffer
  if (m_callback_data.pos > m_audio_out_spec.samples * 2) {
    m_callback_data.buffer.erase(m_callback_data.buffer.begin(),
				 m_callback_data.buffer.begin() + 
				 m_callback_data.pos);
    m_callback_data.pos = 0;
  }

  // Play audio
  for (int i = 0; i < length; i++)
    m_callback_data.buffer.push_back(buf[i]);
  SDL_UnlockAudio();
  SDL_PauseAudio(0);
}

void
AudioPlayer::stop()
{
  SDL_PauseAudio(1);
}
