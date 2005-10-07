#ifndef AUDIOPLAYER_HH
#define AUDIOPLAYER_HH

#include <SDL/SDL.h>
#include <exception>
#include <string>
#include <vector>

/** A class for playing audio files. */
class AudioPlayer {
public:
  
  /** Structure for passing audio data to callback function. */
  struct CallbackData {
    std::vector<short> buffer; //!< The output buffer queue.
    int pos; //!< The position of the next sample to be played.
  };

  /** Create a player. */
  AudioPlayer();

  /** Destroy the player. */
  ~AudioPlayer();

  /** Add a buffer to the queue. */
  void enqueue(short *buf, int length);

  /** Stop playing. */
  void stop();

private:

  CallbackData m_callback_data; //!< The data passed to the callback function.
  SDL_AudioSpec m_audio_out_spec; //!< The audio output format.
};

#endif
