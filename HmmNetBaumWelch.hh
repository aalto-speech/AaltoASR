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

  /** Special pdf_id values used in arcs. */
  enum { EPSILON = -1, FINAL_PDF = -2 };

  static struct {
    double zero(void) { return -1e4; }
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
      double *prob_ptr; //!< Probabilities are in reverse order
      FrameBlock(int s, int e, double *p) : start(s), end(e), prob_ptr(p) { }
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
    int pdf_id; //!< Input symbol indentifying a pdf in the HMM model
    std::string out_str; //!< Output string 
    double score; //!< Score of the arc (should be negative)
    FrameProbs bw_scores; //!< Log likelihoods computed in backward phase
    
    Arc(int id_, int source_, int target_, int pdf_id_, std::string &str_,
        float score_)
      : id(id_), source(source_), target(target_), pdf_id(pdf_id_),
        out_str(str_), score(score_) { }
    ~Arc() { bw_scores.clear(); }
    bool epsilon() { return pdf_id == EPSILON; }
    bool final() { return pdf_id == FINAL_PDF; }
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

  /// Set the pruning thresholds
  void set_pruning_thresholds(double backward, double forward);

  // Segmentator interface
  virtual void open(std::string ref_file);
  virtual void close();
  virtual void set_frame_limits(int first_frame, int last_frame);
  virtual void set_collect_transition_probs(bool collect) { m_collect_transitions = collect; }
  virtual void init_utterance_segmentation(void) { fill_backward_probabilities(); }
  virtual int current_frame(void) { return m_current_frame; }
  virtual bool next_frame(void);
  virtual void reset(void);
  virtual bool eof(void) { return m_eof_flag; }
  virtual const std::vector<Segmentator::StateProbPair>& state_probs(void) { return m_state_prob_pairs; }
  virtual const Segmentator::TransitionMap& transition_probs(void) { return m_transition_info; }

private:
  /** Compute the backward probabilities. */
  void fill_backward_probabilities(void);

  double propagate_node_arcs(int node_id, bool forward,
                             double cur_score, int target_buffer,
                             FeatureVec &fea_vec);
  void add_transition_probabilities(int source_pdf_id, double fw_score,
                                    int next_node_id);
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

  /// Target buffer number in the forward phase
  int m_cur_buffer;

  /// Sum of loglikelihoods of all paths (computed after backward phase)
  double m_sum_total_likelihood;

  /// Table of PDF occupancy probabilities in the forward phase
  std::vector<double> m_pdf_prob;

  /// Table of active PDFs in the forward phase
  std::vector<int> m_active_pdf_table;

  /// A vector which holds the possible states and their probabilities
  std::vector<Segmentator::StateProbPair> m_state_prob_pairs;

  /// true if eof has been detected or frame limits have been reached
  bool m_eof_flag;

  /// true if transitions are to be collected
  bool m_collect_transitions;

  /// Sum of transition likelihoods during one frame
  double m_sum_transition_likelihoods;

  /// A map which holds the information about transitions
  Segmentator::TransitionMap m_transition_info;
};

#endif // HMMNETBAUMWELCH_HH
