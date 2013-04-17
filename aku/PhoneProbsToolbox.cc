#include "PhoneProbsToolbox.hh"

// Use io.h in Visual Studio varjokal 24.3.2010
#ifdef _MSC_VER
#include <io.h>
#include <errno.h>
#include <stdlib.h>
#else
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#endif

#include <fcntl.h>

#include "endian.hh"
#include "io.hh"
#include "str.hh"

#define BYTE unsigned char

// O_BINARY is only defined in Windows
#ifndef O_BINARY
#define O_BINARY 0
#endif

using namespace aku;

void PPToolbox::write_int(FILE *fp, unsigned int i)
{
  BYTE buf[4];

  buf[0] = (i >> 24) & 0xff;
  buf[1] = (i >> 16) & 0xff;
  buf[2] = (i >> 8) & 0xff;
  buf[3] = i & 0xff;
  int ret = fwrite(buf, 4, 1, fp);
  if (ret != 1)
    throw std::string("Write error");
}

void PPToolbox::read_configuration(const std::string &cfgname) {
  gen.load_configuration(io::Stream(cfgname));
}

void PPToolbox::read_models(const std::string &base) {
  model.read_all(base);
}

void PPToolbox::set_clustering(const std::string &clfile_name, double eval_minc, double eval_ming) {
  model.read_clustering(clfile_name);
  model.set_clustering_min_evals(eval_minc, eval_ming);
}

void PPToolbox::generate_to_fd(const int in_fd, const int out_fd, const bool raw_flag) {    
  const int lnabytes=2;
  //io::Stream ofp;
  FILE *ofp;
  const int start_frame=0;

  BYTE buffer[4];
  assert( sizeof(BYTE) == 1 );

  if (model.dim() != gen.dim())
    {
      throw str::fmt(256,
                     "Gaussian dimension is %d but feature dimension is %d.",
                     model.dim(), gen.dim());
    }
  
  // Open files
  gen.open_fd(in_fd, raw_flag);
  ofp=fdopen(out_fd, "wb");

  if(ofp == NULL){
     throw std::string("could not open fd ") + ": " +
      strerror(errno);
  }
  // Write header
  write_int(ofp, model.num_states());
  fputc(lnabytes, ofp);

  // Write the probabilities
  for (int f = start_frame; true ; f++)
    {
      const FeatureVec fea_vec = gen.generate(f);
      if (gen.eof())
	break;

      model.reset_cache();
      model.precompute_likelihoods(fea_vec);
      obs_log_probs.resize(model.num_states());
      double log_normalizer=0;
      for (int i = 0; i < model.num_states(); i++) {
	obs_log_probs[i] = model.state_likelihood(i, fea_vec);
	log_normalizer += obs_log_probs[i];
      }
      if (log_normalizer == 0)
	log_normalizer = 1;
      for (int i = 0; i < (int)obs_log_probs.size(); i++)
	obs_log_probs[i] = util::safe_log(obs_log_probs[i] / log_normalizer);
      

      for (int i = 0; i < model.num_states(); i++)
        {
          if (lnabytes == 4)
	    {
	      BYTE *p = (BYTE*)&obs_log_probs[i];
	      for (int j = 0; j < 4; j++)
		buffer[j] = p[j];
	      if (endian::big)
		endian::convert(buffer, 4);
	    }
          else if (lnabytes == 2)
	    {
	      if (obs_log_probs[i] < -36.008)
		{
		  buffer[0] = 255;
		  buffer[1] = 255;
		}
	      else
		{
		  int temp = (int)(-1820.0 * obs_log_probs[i] + .5);
		  buffer[0] = (BYTE)((temp>>8)&255);
		  buffer[1] = (BYTE)(temp&255);
		}
	    }
          if ((int)fwrite(buffer, sizeof(BYTE), lnabytes, ofp) < lnabytes)
            throw std::string("Write error");
        }
    }
}


void PPToolbox::generate_from_file_to_fd(const std::string &input_name, const int out_fd, const bool raw_flag) {    
  const int lnabytes=2;
  //io::Stream ofp;
  FILE *ofp;
  const int start_frame=0;

  BYTE buffer[4];
  assert( sizeof(BYTE) == 1 );

  if (model.dim() != gen.dim())
    {
      throw str::fmt(256,
                     "Gaussian dimension is %d but feature dimension is %d.",
                     model.dim(), gen.dim());
    }
  
  // Open files
  gen.open(input_name);
  ofp=fdopen(out_fd, "wb");

  // Write header
  write_int(ofp, model.num_states());
  fputc(lnabytes, ofp);

  // Write the probabilities
  for (int f = start_frame; true ; f++)
    {
      const FeatureVec fea_vec = gen.generate(f);
      if (gen.eof())
	break;

      model.reset_cache();
      model.precompute_likelihoods(fea_vec);
      obs_log_probs.resize(model.num_states());
      double log_normalizer=0;
      for (int i = 0; i < model.num_states(); i++) {
	obs_log_probs[i] = model.state_likelihood(i, fea_vec);
	log_normalizer += obs_log_probs[i];
      }
      if (log_normalizer == 0)
	log_normalizer = 1;
      for (int i = 0; i < (int)obs_log_probs.size(); i++)
	obs_log_probs[i] = util::safe_log(obs_log_probs[i] / log_normalizer);
      

      for (int i = 0; i < model.num_states(); i++)
        {
          if (lnabytes == 4)
	    {
	      BYTE *p = (BYTE*)&obs_log_probs[i];
	      for (int j = 0; j < 4; j++)
		buffer[j] = p[j];
	      if (endian::big)
		endian::convert(buffer, 4);
	    }
          else if (lnabytes == 2)
	    {
	      if (obs_log_probs[i] < -36.008)
		{
		  buffer[0] = 255;
		  buffer[1] = 255;
		}
	      else
		{
		  int temp = (int)(-1820.0 * obs_log_probs[i] + .5);
		  buffer[0] = (BYTE)((temp>>8)&255);
		  buffer[1] = (BYTE)(temp&255);
		}
	    }
          if ((int)fwrite(buffer, sizeof(BYTE), lnabytes, ofp) < lnabytes)
            throw std::string("Write error");
        }
    }
}



void PPToolbox::generate(const std::string &input_name, const std::string &output_name, const bool raw_flag) {
  //int in=open(input_name.c_str(), _O_RDONLY | _O_BINARY);
#ifdef _MSC_VER
  int out=open(output_name.c_str(), _O_WRONLY | _O_CREAT | _O_TRUNC | _O_BINARY);
#else
  int out=open(output_name.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_BINARY, 0664);
#endif
  generate_from_file_to_fd(input_name, out, raw_flag);
  gen.close();
  close(out);
};



