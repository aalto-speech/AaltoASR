#ifndef HMMNETBAUMWELCH_HH
#define HMMNETBAUMWELCH_HH

#include "Segmentator.hh"
#include "FeatureGenerator.hh"
#include "HmmSet.hh"

/** A class for generating PDF probabilities for a HMM network.
 * The network is represented in FST format.
 */
class HmmNetBaumWelch : public Segmentator {
public:

  /** Special transition_id values used in arcs. */
  enum { EPSILON = -1, FINAL_TRANSITION = -2 };

  static struct LLType {
    double zero(void) { return -1e10; }
    double one(void) { return 0; }
    double add(double a, double b) { return util::logadd(a,b); }
    double times(double a, double b) { return a + b; }
    double divide(double a, double b) { return a - b; }
  } loglikelihoods;
  
  class FrameProbs {
  public:
    void add_log_prob(int frame, double prob);
    double get_log_prob(int frame);
    void clear(void); //!< Frees the allocated memory
    FrameProbs() : log_prob_table(NULL), num_probs(0), prob_table_size(0) { }
    
  private:
    struct FrameBlock {
      int start;
      int end;

      /// Start index to log_prob_table. Probabilities are in reverse order.
      int buf_start;
      
      FrameBlock(int s, int e, int b) : start(s), end(e), buf_start(b) { }
    };
    std::vector<FrameBlock> frame_blocks; //!< Blocks are in reverse order
    double *log_prob_table;
    int num_probs;
    int prob_table_size;
  };
  
  /** Network arc */
  struct Arc {
    int id;
    int source; //!< Index of the source node 
    int target; //!< Index of the target node

    /// Input symbol identifying a transition in the HMM model
    int transition_id;
    
    std::string out_str; //!< Output string (currently not in use)

    /// Log score of the arc (should be negative). Does not include the
    /// transition probability.
    double score;
    
    FrameProbs bw_scores; //!< Log likelihoods computed in backward phase
    
    Arc(int id_, int source_, int target_, int tr_id_, std::string &str_,
        double score_)
      : id(id_), source(source_), target(target_), transition_id(tr_id_),
        out_str(str_), score(score_) { }
    ~Arc() { bw_scores.clear(); }
    bool epsilon() { return transition_id == EPSILON; }
    bool final() { return transition_id == FINAL_TRANSITION; }
  };
  
  /** Network node */
  struct Node {
    Node(int id_) : id(id_) { prob[0] = loglikelihoods.zero(); prob[1] = loglikelihoods.zero(); }
    int id;
    std::vector<int> in_arcs; //!< Indices of the arcs leading to this node
    std::vector<int> out_arcs; //!< Indices of the arcs leaving this node
    double prob[2]; //!< Log sum of the likelihoods, two buffers
  };

  HmmNetBaumWelch(FeatureGenerator &fea_gen, HmmSet &model);
  virtual ~HmmNetBaumWelch();

  /** Read the network from a file in mitfst ascii format. */
  void read_fst(FILE *file);

  /** Set the pruning thresholds.
   * If the threshold is zero, the current value will not be changed.
   */
  void set_pruning_thresholds(double backward, double forward);

  /// Set the scaling for acoustic log likelihoods
  void set_acoustic_scaling(double scale) { m_acoustic_scale = scale; }

  // Segmentator interface
  virtual void open(std::string ref_file);
  virtual void close();
  virtual void set_frame_limits(int first_frame, int last_frame);
  virtual void set_collect_transition_probs(bool collect) { m_collect_transitions = collect; }
  virtual void init_utterance_segmentation(void);
  virtual int current_frame(void) { return m_current_frame; }
  virtual bool next_frame(void);
  virtual void reset(void);
  virtual bool eof(void) { return m_eof_flag; }
  virtual const std::vector<Segmentator::IndexProbPair>& pdf_probs(void) { return m_pdf_prob_pairs; }
  virtual const std::vector<Segmentator::IndexProbPair>& transition_probs(void) { return m_transition_prob_pairs; }

private:
  /** Computes the backward probabilities. */
  void fill_backward_probabilities(void);

  double propagate_node_arcs(int node_id, bool forward,
                             double cur_score, int target_buffer,
                             FeatureVec &fea_vec);
  double compute_sum_bw_loglikelihoods(int node_id);
  void clear_bw_scores(void);
  
private:
  FeatureGenerator &m_fea_gen;
  HmmSet &m_model;

  std::string m_epsilon_string;
  
  int m_initial_node_id; //!< Index of the unique initial node
  int m_final_node_id; //!< Index of the unique final node
  std::vector<Node> m_nodes; //!< Nodes of the network
  std::vector<Arc> m_arcs; //!< Arcs of the network

  std::vector<int> m_active_node_table[2]; //!< Active nodes (two buffers)

  int m_current_frame; //!< Current frame while computing the probabilities

  /// first frame to be included (0-N)
  int m_first_frame;

  /// last frame to be excluded (0-N); if no limit, m_last_frame = 0
  int m_last_frame;

  /// Beam for pruning the nodes in the backward phase
  double m_backward_beam;

  /// Beam for pruning the arc occupancies in the forward phase
  double m_forward_beam;

  /// Scaling value for acoustic log likelihoods
  double m_acoustic_scale;

  /// Target buffer number in the forward phase
  int m_cur_buffer;

  /// Sum of loglikelihoods of all paths (computed after backward phase)
  double m_sum_total_loglikelihood;

  /// Table of PDF occupancy probabilities in the forward phase
  std::vector<double> m_pdf_prob;

  /// Table of active PDFs in the forward phase
  std::vector<int> m_active_pdf_table;

  /// Table of transition probabilities in the forward phase
  std::vector<double> m_transition_prob;

  /// Table of active transitions in the forward phase
  std::vector<int> m_active_transition_table;

  /// A vector which holds the possible PDFs and their probabilities
  std::vector<Segmentator::IndexProbPair> m_pdf_prob_pairs;

  /// true if eof has been detected or frame limits have been reached
  bool m_eof_flag;

  /// true if transitions are to be collected
  bool m_collect_transitions;

  /// A map which holds the information about transitions
  std::vector<Segmentator::IndexProbPair> m_transition_prob_pairs;
};

#endif // HMMNETBAUMWELCH_HH
