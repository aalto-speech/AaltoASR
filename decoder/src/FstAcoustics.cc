#ifdef _MSC_VER
#include <boost/math/tr1.hpp>
using namespace boost::math::tr1;
#else
#include <math.h>
#endif

#include "FstAcoustics.hh"

FstAcoustics::FstAcoustics(const char *hmm_fname, const char *dur_fname):
  m_frame(0), m_acoustics(nullptr), m_hmm_reader(nullptr)
{
  if (hmm_fname != nullptr) {
    hmm_read(hmm_fname);
  }

  if (dur_fname != nullptr) {
    duration_read(dur_fname);
  }
}

FstAcoustics::~FstAcoustics() {
  if (m_hmm_reader) delete m_hmm_reader;
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void
FstAcoustics::hmm_read(const char *file)
{
  std::ifstream in(file);
  if (!in) {
          std::cerr << "Problems opening " << file << std::endl;
          throw OpenError();
  }
  if (m_hmm_reader) {
    delete m_hmm_reader;
  }
  m_hmm_reader = new NowayHmmReader();
  m_hmm_reader->read(in);
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void FstAcoustics::lna_open(const char *file, int size)
{
        //std::cerr << "Lock held by " << lock_held_by << " " <<1 << std::endl;
        //lock_held_by=1;
        {
#ifdef THREAD_LOCKS
                std::lock_guard<std::mutex> lock(m_fstaio_lock);
#endif
                m_lna_reader.open_file(file, size);
                m_acoustics = &m_lna_reader;
        }
        //lock_held_by=-1;
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void FstAcoustics::lna_open_fd(const int fd, int size)
{
        //std::cerr << "Lock held by " << lock_held_by << " " <<2 << std::endl;
        //lock_held_by=2;
        {
#ifdef THREAD_LOCKS
                std::lock_guard<std::mutex> lock(m_fstaio_lock);
#endif
                m_lna_reader.open_fd(fd, size);
                m_acoustics = &m_lna_reader;
        }
        //lock_held_by=-1;
        //std::cerr << "Lock release 2" << std::endl;
}

// This is a direct copy from Toolbox.cc. FIXME: Code duplication
void FstAcoustics::lna_close()
{
        //std::cerr << "Lock held by " << lock_held_by << " " <<3 << std::endl;
        //lock_held_by=3;
        {
#ifdef THREAD_LOCKS
                std::lock_guard<std::mutex> lock(m_fstaio_lock);
#endif
                m_lna_reader.close();
                m_frame = 0;
        }
        //lock_held_by=-1;
}

void FstAcoustics::duration_read(const char *fname, std::vector<float> *a_table_ptr, std::vector<float> *b_table_ptr) {
  std::ifstream dur_in(fname);
  if (!dur_in) {
          std::cerr << "Problems opening " << fname << std::endl;
          throw OpenError();
  }

  int version;
  float a,b;

  std::vector<float> &a_table = a_table_ptr ? *a_table_ptr : m_a_table;
  std::vector<float> &b_table = b_table_ptr ? *b_table_ptr : m_b_table;

  dur_in >> version;
  if (version!=4)
    throw InvalidFormat();

  int num_states, state_id;
  dur_in >> num_states;
  a_table.resize(num_states);
  b_table.resize(num_states);

  for (int i=0; i<num_states; i++) {
    dur_in >> state_id;
    if (state_id != i) {
      throw InvalidFormat();
    }
    dur_in >> a >> b;
    a_table.push_back(a);
    b_table.push_back(b);
  }
}

float FstAcoustics::duration_logprob(int emission_pdf_idx, int duration) {
  //fprintf(stderr, "Request dur for %d (%d)\n", emission_pdf_idx, duration);
  float a = m_a_table[emission_pdf_idx];
  if (a<=0) {
    return 0.0f;
  }

  float b = m_b_table[emission_pdf_idx];
  float const_term = -a*logf(b)-lgammaf(a);
  return (a-1)*logf(duration)-duration/b+const_term;
}

