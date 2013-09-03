#include "FstSearch.hh"

FstSearch::FstSearch(const char * search_fst_fname, const char * hmm_path, const char * dur_path):
  m_hmm_reader(NULL), m_lna_reader(NULL)
{
  m_fst.read(search_fst_fname);
  m_node_best_token.resize(m_fst.nodes.size());
  m_active_tokens = &m_token_buffer1;

  hmm_read(hmm_path);
  if (dur_path != NULL) {
    duration_read(dur_path);
  }

  m_lna_reader = new LnaReaderCircular;

}

FstSearch::~FstSearch() {
  if (m_hmm_reader) delete m_hmm_reader;
  if (m_lna_reader) delete m_lna_reader;
}


// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void
FstSearch::hmm_read(const char *file)
{
  std::ifstream in(file);
  if (!in)
    throw OpenError();
  if (m_hmm_reader) {
    delete m_hmm_reader;
  }
  m_hmm_reader = new NowayHmmReader();
  m_hmm_map = &(m_hmm_reader->hmm_map());
  m_hmms = &(m_hmm_reader->hmms());
  m_hmm_reader->read(in);
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void FstSearch::duration_read(const char *dur_file)
{
  std::ifstream dur_in(dur_file);
  if (!dur_in)
    throw OpenError();
  m_hmm_reader->read_durations(dur_in);
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void
FstSearch::lna_open(const char *file, int size)
{
  m_lna_reader->open_file(file, size);
  m_acoustics = m_lna_reader;
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void
FstSearch::lna_open_fd(const int fd, int size)
{
  m_lna_reader->open_fd(fd, size);
  m_acoustics = m_lna_reader;
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void
FstSearch::lna_close()
{
  m_lna_reader->close();
}
