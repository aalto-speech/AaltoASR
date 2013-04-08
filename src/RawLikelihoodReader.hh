#ifndef RAWLIKELIHOODREADER_HH
#define RAWLIKELIHOODREADER_HH

#include "Acoustics.hh"

/** A class for reading raw likelihood stream. */
class RawLikelihoodReader : public Acoustics {
public:

  struct IOException { };

  /** Create a reader. */
  RawLikelihoodReader();

  /** Create a reader with given number of likelihoods. */
  RawLikelihoodReader(int num_likelihoods);

  /** Set the number of likelihoods. */
  void set_num_likelihoods(int num_likelihoods);

  /** Attach to a file. */
  void open(FILE *file);

  /** Read frames up the given frame. */
  virtual bool go_to(int frame);

private:
  std::vector<float> m_likelihoods; //!< Storage for likelihoods
  FILE *m_file; //!< The file to read the likelihoods from
  int m_frame; //!< The current frame
};

#endif /* RAWLIKELIHOODREADER_HH */
