// Fairly compact prefix tree represantation for n-gram language model
#ifndef TREEGRAM_HH
#define TREEGRAM_HH

#include <cstddef>  // NULL
#include "NGram.hh"

class TreeGram : public NGram {
public:
  struct Node {
    Node() : word(-1), log_prob(0), back_off(0), child_index(-1) {}
    Node(int word, float log_prob, float back_off, int child_index)
      : word(word), log_prob(log_prob), back_off(back_off), 
	child_index(child_index) {}
    int word;
    float log_prob;
    float back_off;
    int child_index;
  };

  struct ReadError : public std::exception {
    virtual const char *what() const throw()
      { return "TreeGram: read error"; }
  };

  class Iterator {
  public:
    Iterator(TreeGram *gram = NULL);
    void reset(TreeGram *gram);

    // Move to the next node in depth-first order
    bool next();

    // Move to the next node on the given order
    bool next_order(int order);

    // Return the node from the index stack. (default: the last one)
    Node &node(int order = 0);

    // Order of the current node (1 ... n)
    int order() { return m_index_stack.size(); }

    // Move to within current context (default: to the next word)
    bool move_in_context(int delta = 1);

    // Come back up to previous order
    bool up();

    // Dive down to first child
    bool down();

    bool has_children();

    friend class TreeGram;
  private:
    TreeGram *m_gram;
    std::vector<int> m_index_stack;
  };

  void reserve_nodes(int nodes); 

  /// \brief Adds a new gram to the language model.
  ///
  /// The grams must be inserted in sorted order.  The only exception
  /// is the OOV 1-gram, which can be updated any time.  It exists by
  /// default with very small log-prob and zero back-off.
  void add_gram(const Gram &gram, float log_prob, float back_off, bool add_missing_unigrams=false);

  /// \brief Reads a language model file.
  ///
  /// \param binary If false, the file is expected to be in ARPA file format.
  ///
  void read(FILE *file, bool binary=false);

  void write(FILE *file, bool binary=false);
  void write_real(FILE *file, bool reflip);

  float log_prob_bo(const Gram &gram); // Keep this version lean and mean
  float log_prob_bo_cl(const Gram &gram); // Clustered backoff
  float log_prob_i(const Gram &gram); // Interpolated
  float log_prob_i_cl(const Gram &gram); //Interpolated backoff

  inline float log_prob_bo(const std::vector<int> &gram) {
    Gram g(gram.size());
    for (size_t i=0;i<gram.size();i++) g[i]=gram[i];
    return(log_prob_bo(g));
  }

  inline float log_prob_bo_cl(const std::vector<int> &gram) {
    Gram g(gram.size());
    for (size_t i=0;i<gram.size();i++) g[i]=gram[i];
    return(log_prob_bo_cl(g));
  }

  inline float log_prob_i(const std::vector<int> &gram){
    Gram g(gram.size());
    for (size_t i=0;i<gram.size();i++) g[i]=gram[i];
    return(log_prob_i(g));
  }

  inline float log_prob_i_cl(const std::vector<int> &gram){
    Gram g(gram.size());
    for (size_t i=0;i<gram.size();i++) g[i]=gram[i];
    return(log_prob_i_cl(g));
  }

  inline float log_prob_bo(const std::vector<unsigned short> &gram) {
    Gram g(gram.size());
    for (size_t i=0;i<gram.size();i++) g[i]=gram[i];
    return(log_prob_bo(g));
  }

  inline float log_prob_bo_cl(const std::vector<unsigned short> &gram) {
    Gram g(gram.size());
    for (size_t i=0;i<gram.size();i++) g[i]=gram[i];
    return(log_prob_bo_cl(g));
  }

  inline float log_prob_i(const std::vector<unsigned short> &gram){
    Gram g(gram.size());
    for (size_t i=0;i<gram.size();i++) g[i]=gram[i];
    return(log_prob_i(g));
  }

  inline float log_prob_i_cl(const std::vector<unsigned short> &gram){
    Gram g(gram.size());
    for (size_t i=0;i<gram.size();i++) g[i]=gram[i];
    return(log_prob_i_cl(g));
  }

  int gram_count(int order) { return m_order_count.at(order-1); }

  /* Don't use this function, unles you really need to*/
  int find_child(int word, int node_index);

  // Returns an iterator for given gram.
  Iterator iterator(const Gram &gram);

  /// \brief Computes bigram probabilities for every word pair
  /// with context "prev_word_id".
  ///
  /// Used for LM lookahead in the recognizer.
  ///
  /// \param result_buffer Will have bigram probabilities when available, unigram
  /// probabilities for other words, and OOVs will have logprob
  /// \a prev_word_id's backoff weight - 99.
  ///
  void fetch_bigram_list(int prev_word_id,
                         std::vector<float> &result_buffer);

  /// \brief Computes trigram probabilities for every word triplet
  /// with context "w1 w2".
  ///
  /// Used for LM lookahead in the recognizer.
  ///
  void fetch_trigram_list(int w1, int w2,
                          std::vector<float> &result_buffer);

  void print_debuglist();
  void finalize(bool add_missing_unigrams=false);
  void convert_to_backoff();

private:
  int binary_search(int word, int first, int last);
  void print_gram(FILE *file, const Gram &gram);
  void find_path(const Gram &gram);
  void check_order(const Gram &gram, bool add_missing_unigrams=false);
  void flip_endian();
  void fetch_gram(const Gram &gram, int first);

  std::vector<int> m_order_count;	// number of grams in each order
  std::vector<Node> m_nodes;		// storage for the nodes
  std::vector<int> m_fetch_stack;	// indices of the gram requested
  //int m_last_order;			// order of the last hit

  // For creating the model
  std::vector<int> m_insert_stack;	// indices of the last gram inserted
  Gram m_last_gram;			// the last ngram added to the model
};

#endif /* TREEGRAM_HH */
