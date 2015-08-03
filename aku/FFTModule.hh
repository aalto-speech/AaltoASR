#ifndef FFTMODULE_HH
#define FFTMODULE_HH

#include <vector>

#ifdef KISS_FFT
#include "kiss_fftr.h"
#else // Use FFTW
#include <fftw3.h>
#endif

#include "FeatureModule.hh"


namespace aku {

class FFTModule : public FeatureModule {
public:
  FFTModule();
  virtual ~FFTModule();
  static const char *type_str() { return "fft"; }

private:
  virtual void get_module_config(ModuleConfig &config);
  virtual void set_module_config(const ModuleConfig &config);
  virtual void generate(int frame);

  int m_magnitude; //!< If nonzero, compute magnitude spectrum instead of power
  int m_log; //!< If nonzero, take logarithms of the output

  std::vector<float> m_hamming_window;
#ifdef KISS_FFT
  kiss_fftr_cfg m_coeffs;
  kiss_fft_scalar *m_kiss_fft_datain;
  kiss_fft_cpx *m_kiss_fft_dataout;
#else
  fftw_plan m_coeffs;
  std::vector<double> m_fftw_datain;
  std::vector<double> m_fftw_dataout;
#endif
};

}

#endif
