#ifndef BASEFEAMODULE_HH
#define BASEFEAMODULE_HH

#include "FeatureModule.hh"


namespace aku {

// Interface for modules that generate features from a file
class BaseFeaModule : public FeatureModule {
public:
  virtual void set_fname(const char *fname) = 0;
  virtual void set_file(FILE *fp, bool stream=false) = 0;
  virtual void discard_file(void) = 0;

  // Note: eof() may return false for a frame that is past the real EOF
  // if at() for that frame has not yet been called.
  virtual bool eof(int frame) = 0;

  virtual int sample_rate(void) = 0;
  virtual float frame_rate(void) = 0;

  virtual int last_frame(void) = 0;

  virtual void add_source(FeatureModule *source) 
  { throw std::string("base module FFT can not have sources"); }
};

}

#endif
