#ifndef HMMNETBAUMWELCH_HH
#define HMMNETBAUMWELCH_HH

#include "Segmentator.hh"
#include "FeatureGenerator.hh"
#include "HmmSet.hh"

/** A class for generating state probabilities for a HMM network.
 * The network is represented in FST format.
 */
class HmmNetBaumWelch : public Segmentator {
public:

  /** Special transition_id values used in arcs. */
  enum { EPSILON = -1, FINAL_TRANSITION = -2 };

  enum { MODE_BAUM_WELCH = 1, MODE_VITERBI = 2, MODE_EXTENDED_VITERBI = 3};
  
  static struct LLType {
    double zero(void) { return -1e15; }
    double one(void) { return 0; }
    double add(double a, double b) { return util::logadd(a,b); }
    double times(double a, double b) { return a + b; }
    double divide(double a, double b) { return a - b; }
  } loglikelihoods;
  
  class FrameScores {
  public:
    void set_score(int frame, double score);
    void set_new_score(int frame, double score);
    double get_score(int frame);
    void clear(void); //!< Frees the allocated memory
    FrameScores() : score_table(NULL), num_scores(0), score_table_size(0) { }
    
  private:
    struct FrameBlock {
      int start;
      int end;

      /// Start index to score_table. Probabilities are in reverse order.
      int buf_start;
      
      FrameBlock(int s, int e, int b) : start(s), end(e), buf_start(b) { }
    };
    std::vector<FrameBlock> frame_blocks; //!< Blocks are in reverse order
    double *score_table;
    int num_scores;
    int score_table_size;
  };
  
  /** Network arc */
  struct Arc {
    int id;
    int source; //!< Index of the source node 
    int target; //!< Index of the target node

    /// Input symbol identifying a transition in the HMM model
    int transition_id;
    
    std::string label; //!< Arc label

    /// Log score of the arc (should be negative). Does not include the
    /// transition probability.
    double score;
    
    FrameScores bw_scores; //!< Log likelihoods computed in backward phase

    /// Application specific probability-type data. This is queried in
    /// the backward phase using the \ref CustomDataQuery interface if
    /// it has been defined using \ref set_custom_data_callback.
    FrameScores bw_custom_data;
    
    Arc(int id_, int source_, int target_, int tr_id_, std::string &str_,
        double score_)
      : id(id_), source(source_), target(target_), transition_id(tr_id_),
        label(str_), score(score_) { }
    ~Arc() { bw_scores.clear(); }
    bool epsilon() { return transition_id == EPSILON; }
    bool final() { return transition_id == FINAL_TRANSITION; }
  };
  
  /** Network node */
  struct Node {
    Node(int id_) : id(id_) { log_prob[0] = loglikelihoods.zero(); log_prob[1] = loglikelihoods.zero(); num_epsilon_out = 0; }
    int id;
    std::vector<int> in_arcs; //!< Indices of the arcs leading to this node
    std::vector<int> out_arcs; //!< Indices of the arcs leaving this node

    // If nonzero, all out transitions (not counting the self transition) are
    // epsilon transitions and this is their count. Otherwise there are no
    // epsilon out transitions.
    int num_epsilon_out;
    
    double log_prob[2]; //!< Log sum of the likelihoods, two buffers

    /// Average of the custom data over all sequences starting/ending
    /// at this node
    double custom_score[2];
  };


  /** Information on arcs traversed at last frame during forward phase */
  struct TraversedArc {
    int arc_id; //!< Index in m_arcs
    double score; //!< Current log probability
    double custom_score; //! Average custom data
    TraversedArc(int i, double s, double c) : arc_id(i), score(s), custom_score(c) { }
  };

  /** Arc information used in \ref fill_arc_info */
  struct ArcInfo {
    std::string label; //!< Arc label
    double prob; //!< Probability of arc traversal during last frame
    int pdf_index; //!< PDF index associated with this arc
    double custom_score; //! Average custom score over paths that pass this arc
    ArcInfo(std::string &l, double p, int pdf, double c) : label(l), prob(p), pdf_index(pdf), custom_score(c) { }
  };


  /** Callback interface for querying custom data values */
  class CustomDataQuery {
  public:
    virtual ~CustomDataQuery() { }
    virtual double custom_data_value(int frame, Arc &arc) = 0;
  };
  

  HmmNetBaumWelch(FeatureGenerator &fea_gen, HmmSet &model);
  virtual ~HmmNetBaumWelch();

  /** Read the network from a file in mitfst ascii format. */
  void read_fst(FILE *file);

  /** Set the pruning thresholds.
   * If the threshold is zero, the current value will not be changed.
   */
  void set_pruning_thresholds(double backward, double forward);

  double get_backward_beam(void) { return m_backward_beam; }
  double get_forward_beam(void) { return m_forward_beam; }

  /// Set the segmentation mode
  void set_mode(int mode) { m_mode = mode; }

  /// Set the scaling for acoustic log likelihoods
  void set_acoustic_scaling(double scale) { m_acoustic_scale = scale; }

  /// Set the custom data callback interface and enable its usage
  void set_custom_data_callback(CustomDataQuery *callback) { m_custom_data_callback = callback; }

  /// Fills in information about traversed arcs
  void fill_arc_info(std::vector<ArcInfo> &traversed_arcs);

  /// \returns Average of the sum of the custom data over all paths
  double get_total_custom_score(void) { return m_sum_total_custom_score; }

  // Segmentator interface
  virtual void open(std::string ref_file);
  virtual void close();
  virtual void set_frame_limits(int first_frame, int last_frame);
  virtual void set_collect_transition_probs(bool collect) { m_collect_transitions = collect; }
  virtual bool init_utterance_segmentation(void);
  virtual int current_frame(void) { return m_current_frame; }
  virtual bool next_frame(void);
  virtual void reset(void);
  virtual bool eof(void) { return m_eof_flag; }
  virtual bool computes_total_log_likelihood(void) { return true; }
  virtual double get_total_log_likelihood(void) { return m_sum_total_loglikelihood; }
  virtual const std::vector<Segmentator::IndexProbPair>& pdf_probs(void) { return m_pdf_prob_pairs; }
  virtual const std::vector<Segmentator::IndexProbPair>& transition_probs(void) { return m_transition_prob_pairs; }
  virtual const std::string& highest_prob_label(void) { return m_most_probable_label; }

private:

  /** Checks the network structure and fills the epsilon flags for nodes.
   * Throws an exception if invalid structure is detected.
   */
  void check_network_structure(void);
  
  /** Computes the backward probabilities.
   * \return true if successful, false if backward beam should be increased. */
  bool fill_backward_probabilities(void);

  double backward_propagate_node_arcs(int node_id, double cur_score,
                                      double cur_custom_score,
                                      int target_buffer, bool epsilons,
                                      FeatureVec &fea_vec,double ref_log_prob);
  void forward_propagate_node_arcs(int node_id, double cur_score,
                                   double cur_custom_score,
                                   int target_buffer, bool epsilons,
                                   FeatureVec &fea_vec);
  void fill_arc_scores(int arc_id, FeatureVec &fea_vec,
                       double cur_custom_score,
                       double *arc_score, double *arc_custom_score);
  void update_node_custom_score(int node_id, int target_buffer,
                                double old_log_prob, double new_log_prob,
                                double new_custom_score);
  void compute_total_bw_scores(void);
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

  /// Segmenting mode
  int m_mode;


  /// A pointer to custom data callback interface (NULL if not in use)
  CustomDataQuery *m_custom_data_callback;
  
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

  /// Average of the custom data over all paths
  double m_sum_total_custom_score;

  /// Table of PDF occupancy probabilities in the forward phase
  std::vector<double> m_pdf_prob;

  /// Table of active PDFs in the forward phase
  std::vector<int> m_active_pdf_table;

  /// Table of transition probabilities in the forward phase
  std::vector<double> m_transition_prob;

  /// Table of active transitions in the forward phase
  std::vector<int> m_active_transition_table;

  /// Table of active arcs in the forward phase
  std::vector<TraversedArc> m_active_arcs;

  /// A vector which holds the possible PDFs and their probabilities
  std::vector<Segmentator::IndexProbPair> m_pdf_prob_pairs;

  /// String containing the current most probable arc label
  std::string m_most_probable_label;

  /// true if eof has been detected or frame limits have been reached
  bool m_eof_flag;

  /// true if transitions are to be collected
  bool m_collect_transitions;

  /// A map which holds the information about transitions
  std::vector<Segmentator::IndexProbPair> m_transition_prob_pairs;
};

#endif // HMMNETBAUMWELCH_HH
