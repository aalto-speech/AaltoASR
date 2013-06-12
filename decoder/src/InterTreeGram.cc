#include "io.hh"
#include "def.hh"
#include "TreeGramArpaReader.hh"
#include "InterTreeGram.hh"

InterTreeGram::InterTreeGram(const std::vector< std::string > lm_names, const std::vector<float> coeffs) {
  if (lm_names.size() != coeffs.size()) {
    fprintf(stderr, "InterTreeGram::InterTreeGram: There must be as many interpolation coeffs as there are LMs. Exit.\n");
    exit(1);
  }

  float coeff_sum=0.0;
  for(std::vector<float>::const_iterator j=coeffs.begin();j!=coeffs.end();++j) {
    coeff_sum += *j;
  }

  if (coeff_sum < 0.99 || coeff_sum>1.01) {
    fprintf(stderr, "InterTreeGram::InterTreeGram: Interpolation coeffs must sum to 1 (!=%f). Exit.\n", coeff_sum);
    exit(1);
  }
  m_coeffs = coeffs;

  // Combine vocab from all models
  for ( std::vector<std::string>::const_iterator it = lm_names.begin(); it != lm_names.end(); it ++) {
    ArpaReader areader(this);
    io::Stream lm_in(*it, "r");
    
    std::string line;
    bool dummy;
    areader.read_header(lm_in.file, dummy, line);

    std::vector<int> tmp_gram(1);
    float log_prob, back_off;
    // This adds all words to vocab and discards the other information
    while( areader.next_gram(lm_in.file, line, tmp_gram, log_prob, back_off) && tmp_gram.size()==1) {}
  }

  // Initialize all models
  for ( std::vector<std::string>::const_iterator it = lm_names.begin(); it != lm_names.end(); it ++) {
    io::Stream lm_in(*it, "r");
    TreeGram *lm = new TreeGram;

    // Copy vocab;
    int real_num_words = num_words();
    copy_vocab_to(*lm);
    assert(lm->num_words() == real_num_words);
    for (int i=0; i<num_words(); i++) {
      assert(lm->word(i)==word(i));
    }

    // Read n_grams
    TreeGramArpaReader tgar;
    tgar.read(lm_in.file, lm, true);

    //io::Stream lm_out("whaat", "w");
    //TreeGramArpaReader tga2;
    //tga2.write(lm_out.file, lm);
    
    m_models.push_back(lm);
    assert(lm->num_words() == real_num_words);

    if (lm->order()>m_order) {
      m_order = lm->order();
    }
  }
}

InterTreeGram::~InterTreeGram(void) {
  for (std::vector<TreeGram *>::iterator j=m_models.begin();j!=m_models.end();++j) {
    delete *j;
  }
}

float InterTreeGram::log_prob(const Gram &gram) {
  double prob=0.0;
  for (int i=0; i<m_models.size(); i++) {
    prob += m_coeffs[i] * pow(10, m_models[i]->log_prob(gram));
  }
  return safelogprob(prob);
}

void InterTreeGram::test_write(std::string fname, int idx) {
  io::Stream lm_out(fname, "w");
  TreeGramArpaReader tga;
  tga.write(lm_out.file, m_models[idx]);
}

void InterTreeGram::fetch_bigram_list(int prev_word_id, 
                                      std::vector<float> &result_buffer) {
  /* This may be too slow for real use */

  // Zero the result buffer
  result_buffer.resize(m_words.size());
  for (std::vector<float>::iterator it=result_buffer.begin(); it!=result_buffer.end(); ++it) {
    *it = 0.0f;
  }

  // Accumulate probs
  std::vector<float> cresbuf(result_buffer.size());
  for (int i=0; i<m_models.size(); i++) {
    m_models[i]->fetch_bigram_list(prev_word_id, cresbuf);
    for (int j=0; j<result_buffer.size(); j++) {
      result_buffer[j] += m_coeffs[i] * pow(10, cresbuf[j]);
    }
  }

  // logprob results
  for (std::vector<float>::iterator it=result_buffer.begin(); it!=result_buffer.end(); ++it) {
    *it = safelogprob(*it);
  }
}
