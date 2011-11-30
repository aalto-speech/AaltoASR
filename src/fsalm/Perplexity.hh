#ifndef PERPLEXITY_HH
#define PERPLEXITY_HH

#include <cstddef>  // NULL
#include "lm/LM.hh"
#include "misc/macros.hh"

namespace rec {
  
  /** Compute perplexity or cross-entropy of a language model. */
  class Perplexity {
  public:
    
    /** Create a new perplexity computer with a language model. */
    Perplexity(LM *lm = NULL)
    {
      opt.word_boundary_str = "<w>";
      opt.unk_str = "";
      reset(lm);
    }

    /** Reset perplexity counters. */
    void reset(LM *lm = NULL)
    {
      m_start_pending = true;
      m_score = 0;
      m_num_symbols = 0;
      m_num_words = 0;
      m_num_sentences = 0;
      m_lm_node_id = -1;
      if (lm != NULL)
        m_lm = lm;
    }

    /** Number of symbols. */
    int num_symbols() const
    {
      return m_num_symbols;
    }

    /** Number of words. */
    int num_words() const
    {
      return m_num_words;
    }

    /** Number of sentences. */
    int num_sentences() const
    {
      return m_num_sentences;
    }

    /** Log-probability of the test data. */
    double score() const
    {
      return m_score;
    }

    /** Compute cross-entropy in bits assuming that score is in
     * log10. 
     * \throw bit::invalid_call if no words yet
     */
    float cross_entropy_per_word() const
    {
      if (m_num_words == 0)
        throw Error("Perplexity::cross_entropy_per_word() no words yet");
      return m_score * 3.3219280949 / m_num_words;
    }

    /** Add a symbol to eval set.
     *
     * \param symbol = symbol to add
     * \return log10-probability of the word
     *
     * \throw bit::invalid_argument if sentence start was missing or
     * sentence start was given before sentence end, or unknown symbol
     * was given
     */
    float add_symbol(const Str &symbol_str)
    {
      int symbol = m_lm->symbol_map().index_nothrow(symbol_str);
      bool unk = false;
      if (symbol < 0) {
        if (opt.unk_str.empty()) 
          throw bit::invalid_argument(
            Str("bit::Perplexity::add_symbol(): invalid symbol \"") +
            symbol_str + "\"");
        symbol = m_lm->symbol_map().index_nothrow(opt.unk_str);
        if (symbol < 0) {
          throw bit::invalid_argument(
            Str("bit::Perplexity::add_symbol(): invalid unk symbol \"")
            + opt.unk_str + "\"");
        }
        unk = true;
      }
      
      if (symbol == m_lm->start_symbol() && !m_start_pending)
        throw bit::invalid_argument(
          "bit::Perplexity::add_symbol() unexpected start symbol");
      if (symbol != m_lm->start_symbol() && m_start_pending)
        throw bit::invalid_argument(
          "bit::Perplexity::add_symbol() expected start symbol but got \"" +
          symbol_str + "\"");

      float score = 0;

      if (symbol == m_lm->start_symbol()) {
        m_start_pending = false;
        assert(m_lm_node_id == -1);
        m_lm_node_id = m_lm->initial_node_id();
        score = m_lm->final_score();
        m_score += score;
        return score;
      }

      m_lm_node_id = m_lm->walk(m_lm_node_id, symbol, &score);

      if (symbol != m_lm->end_symbol() && !unk)
        m_num_symbols++;

      if (!unk)
        m_score += score;

      if (symbol_str == opt.word_boundary_str)
        m_num_words++;

      if (symbol == m_lm->end_symbol()) {
        m_start_pending = true;
        m_lm_node_id = -1;
        m_num_sentences++;
        m_num_words--;
      }

      return score;
    }

    /** Evaluate all sentences in a text string. 
     *
     * \param sentence = symbols of the text separated by white space
     *
     * \param out = if non-null, symbols and corresponding
     * probabilities are written in the string
     *
     * \return log probability of the sentence
     */
    float eval(Str text, Str *out = NULL)
    {
      double sum = 0;
      StrVec symbols = str::split(text, " \t", true);
      FOR(i, symbols) {
        float score = add_symbol(symbols[i]);
        if (out != NULL) {
          if (i > 0)
            out->append(" ");
          out->append(symbols[i]);
          out->append(str::fmt(64, " %g", score));
        }
        sum += score;
      }
      return sum;
    }

    /** As above but return the output string instead of probability. */
    Str eval_str(Str text)
    {
      Str out;
      eval(text, &out);
      return out;
    }

    /** Evaluate file containing one sentence per line.  Start and end
     * symbols are inserted automatically if missing.
     *
     * \param input = file to read the sentences from
     * \param ouput = file to output probability stream
     */
    void eval_file(FILE *input, FILE *output = NULL)
    {
      Str line;
      StrVec symbols;
      while (str::read_line(line, input, true)) {
        str::clean(line);
        if (line.empty())
          continue;
        symbols = str::split(line, " \t", true);
        assert(!symbols.empty());
        if (symbols[0] != m_lm->start_str)
          symbols.insert(symbols.begin(), m_lm->start_str);
        if (symbols.back() != m_lm->end_str)
          symbols.push_back(m_lm->end_str);
        FOR(i, symbols) {
          float log10prob = add_symbol(symbols[i]);
          float prob = 0;
          if (log10prob > -20)
            prob = pow(10, log10prob);
          if (output != NULL)
            fprintf(output, "%-10g %-10g %s\n", log10prob, prob, symbols[i].c_str());
        }
      }
    }

  public:
    struct {
      /** Symbol used for computing number of words.  It is assumed
       * that word boundary comes always after sentence start and
       * before sentence end. */
      Str word_boundary_str;

      /** Symbol used for unknown word.  Empty if unk is not used. */
      Str unk_str;
    } opt;

  private:

    /** Are we waiting the start of sentence? */
    bool m_start_pending;

    /** Score of the test data. */
    double m_score;

    /** Number of symbols in the test data (not including sentence starts) */
    int m_num_symbols;

    /** Number of words in the test data. */
    int m_num_words;

    /** Number of sentences in the test data. */
    int m_num_sentences;

    /** The language model used for computing the perplexity. */
    const LM *m_lm;

    /** Iterator for walking in the language model. */
    int m_lm_node_id;

  };

};

#endif /* PERPLEXITY_HH */
