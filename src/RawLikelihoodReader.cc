#include "RawLikelihoodReader.hh"

RawLikelihoodReader::RawLikelihoodReader()
  : m_frame(-1)
{
  m_num_models = 0;
  m_log_prob = NULL;
}

RawLikelihoodReader::RawLikelihoodReader(int num_likelihoods)
  : m_frame(-1),
    m_likelihoods(num_likelihoods)
{
  m_num_models = num_likelihoods;
  m_log_prob = &m_likelihoods[0];
}

void
RawLikelihoodReader::set_num_likelihoods(int num_likelihoods)
{
  m_num_models = num_likelihoods;
  m_likelihoods.resize(num_likelihoods);
  m_log_prob = &m_likelihoods[0];
}

void
RawLikelihoodReader::open(FILE *file)
{
  m_frame = 0;
  m_file = file;
}

bool
RawLikelihoodReader::go_to(int frame)
{
  while (m_frame < frame) {
    int num_read = fread(m_log_prob, sizeof(float), m_num_models, m_file);

    // End of file?
    if (num_read == 0)
      return false;

    // Short read?
    if (num_read != m_num_models) {
      fprintf(stderr, "RawLikelihoodReader::go_to(): fread() failed\n");
      throw IOException();
    }
    m_frame++;
  }
  return true;
}
