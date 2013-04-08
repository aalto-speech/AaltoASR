#ifndef HMMNETBAUMWELCH_HH
#define HMMNETBAUMWELCH_HH

#include <set>

#include "Segmentator.hh"
#include "FeatureGenerator.hh"
#include "HmmSet.hh"

namespace aku {

/** A class for segmenting a HMM network.
 *
 * This class uses two network representations:
 *   1) Search network - this is the input HMM network constructed to
 *      represent the training utterance. The network is stored in FST
 *      format. It may contain epsilon arcs and hierarchically
 *      structured arc labels. However, there are some requirements:
 *      - The lowest hierarchy level represents HMM transition indices,
 *        which define the emitting states. It is called the physical level.
 *      - The higher hierarchy levels represent logical arcs that group a
 *        number of physical arcs together (even in different branches)
 *      - The search network is loaded from an FST-file, in MIT FST ASCII
 *        format. The transitions need to have labels as follows:
 *        - Input symbol: transition_index;logical_arc1;...;logical_arcN
 *         - Output symbol: word/phoneme
 *         - OR: Only input symbol #label (epsilon arc, label is usually
 *           a word or a phoneme)
 *      - If there are multiple out-arcs from a node the node may not
 *        have a self-transition
 *      - Except for the physical level, arc label names must
 *        have a postfix # if that logical arc does not continue after
 *        that arc, that is, there is a logical arc which is "connected"
 *        to the target node.
 *
 *      Note that if segmentation mode MODE_MULTIPATH_VITERBI is used,
 *      then for any path segment without branches (self-transitions
 *      excluded) only a single permutation of self-transitions is used,
 *      one that maximizes the summed loglikelihood of the paths containing
 *      that particular path segment. This means that the search network
 *      affects the segmentation: If several paths pass through the same
 *      node, the segmentation up to that node is optimized with respect
 *      to the summed loglikelihood of the continuation paths. If instead
 *      the paths were presented as separate path segments, more
 *      variations in the segmentations would be possible.
 *
 *      In MODE_MULTIPATH_VITERBI, the arcs that present the self- and
 *      out-transition from the same HMM model are identified based
 *      on the first-level logical arc label. That is, among arcs sharing
 *      the first-level logical arc, only the best transition is traversed.
 *      Epsilon arcs are always propagated without restrictions, even
 *      though they would share the same logical arc (FIXME?).
 *
 *   2) Segmented lattice - this is the result from the forward-backward
 *      (actually backward-forward in this implementation) algorithm.
 *      The result can be browsed using the Segmentator interface, or
 *      the segmented lattice can be used directly. Segmented lattice
 *      does not contain epsilon-arcs at the physical level, but logical
 *      arcs do exist. Each node in the segmented lattice has a fixed
 *      entrance frame number. Therefore different segmentations are
 *      represented as multiple segmented arcs referring to the same
 *      network arc, leading to different segmented nodes with different
 *      entrance frame numbers. The nodes in segmented lattice do not
 *      correspond to nodes in the search network, only the arcs
 *      have a counterpart in the search network.
 *
 *      Segmented lattice object (SegmentedLattice) contains all the
 *      necessary information for most algorithms. Most
 *      ready-implemented operations for the segmented lattices are
 *      therefore part of SegmentedLattice class. One notable
 *      exception is the extraction of a higher level lattice which
 *      requires additional parent relationship information from the
 *      search network, and is therefore implemented in the
 *      HmmNetBaumWelch class.
 */
class HmmNetBaumWelch : public Segmentator {

public:
  /******************* Public type definitions *******************/

  /// Special transitions
  enum { EPSILON = -1 };

  /** Segmentation modes */
  enum { MODE_VITERBI = 1, MODE_MULTIPATH_VITERBI = 2, MODE_BAUM_WELCH = 3 };

  /** Custom score combination modes. Custom path scores are obtained by
   * summing the custom scores of all the arcs in the paths. The combination
   * mode determines what will be the combined path score of the arc.
   * CUSTOM AVG  - Compute the average of custom path scores,
   *               using arc probabilities
   * CUSTOM_SUM  - Compute the sum of custom path scores,
   *               without arc probabilities
   * CUSTOM_MAX  - Compute the maximum of custom path scores for each arc
   */
  enum { CUSTOM_AVG = 1, CUSTOM_SUM = 2, CUSTOM_MAX = 3 };

  /// Loglikelihood operations
  static struct LLType {
    double zero(void) { return -1e15; }
    double one(void) { return 0; }
    double plus(double a, double b) { return util::logadd(a,b); }
    double times(double a, double b) { return a + b; }
    double divide(double a, double b) { return a - b; }
  } loglikelihoods;

  
  class SegmentedLattice; // Forward declaration

  /** Callback interface for querying custom scores */
  class CustomScoreQuery {
  public:
    virtual ~CustomScoreQuery() { }

    /** Returns the custom score for the given arc
     * \param sl Segmented lattice representing the current segmentation
     * \param arc_index Arc index of sl->arcs
     */
    virtual double custom_score(SegmentedLattice const *sl, int arc_index)=0;
  };


  /// Class for storing backward phase scores
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

      /// Start index to log_prob_table. Probabilities are in reverse order.
      int buf_start;
      
      FrameBlock(int s, int e, int b) : start(s), end(e), buf_start(b) { }
    };
    std::vector<FrameBlock> frame_blocks; //!< Blocks are in reverse order
    double *score_table;
    int num_scores;
    int score_table_size;
  };


  /** Logical arc that groups physical arcs or other logical arcs */
  struct LogicalArc {
    int level; // Logical arc level, > 0
    int parent_arc; // Parent logical arc, index to m_logical_arcs (or -1)

    std::string label; //!< Arc label (from current level upwards)

    LogicalArc(int level_, int parent_, std::string &label_) : level(level_), parent_arc(parent_), label(label_) { }
  };
  
  /** Search network arc */
  struct Arc {
    int source; //!< Index of the source node 
    int target; //!< Index of the target node

    /** Index to \ref m_logical_arcs for parent arc in the next
     * hierarchy level (or -1 if no parents) */
    int parent_arc;
    
    /// Index of the transition in the HMM model
    int transition_index;
    
    std::string label; //!< Arc label

    /** Static log score of the arc (should be negative). Does not include the
     * transition probability. */
    double static_score;

    /// Log likelihoods computed in the backward phase. For level == 0 only.
    FrameScores bw_scores;
    
    Arc(int source_, int target_, int parent_,
        int tr_id_, const std::string &label_, double score_)
      : source(source_), target(target_), parent_arc(parent_),
        transition_index(tr_id_), label(label_), static_score(score_) { }
    ~Arc() { bw_scores.clear(); }
    bool epsilon() { return transition_index == EPSILON; }
    bool self_transition() { return source == target; }
  };
  
  /** Search network node */
  struct Node {
    int id;
    std::vector<int> in_arcs; //!< Indices of the arcs leading to this node
    std::vector<int> out_arcs; //!< Indices of the arcs leaving this node

    bool self_transition; //!< true if the node has a self-transition

    Node(int id_) : id(id_) { self_transition = false; }
  };


  /** Segmented lattice arc */
  struct SegmentedArc {
    //!< Internal data: Index to either m_arcs or m_logical_arcs
    int net_arc_id;

    std::string label; //!< Label of the arc

    /** HMM transition index. Valid only if the arc belongs to a
     * frame lattice.
     */
    int transition_index;

    int source_node; //!< SegmentedNode id
    int target_node; //!< SegmentedNode id

    /// Log score of this arc
    double arc_score;

    /** Acoustic score of the arc. At the moment only valid for frame lattices,
     * but could be implemented for others as well
     */
    double arc_acoustic_score;
    
    /// Log sum of the scores of all the paths which contain this arc
    double total_score;
    
    /** Application specific score for this arc.
     * Note: Only available if custom scores were computed on the current
     * level. If custom scores are propagated, only path scores are filled
     * for the frame level lattice.
     */
    double custom_score;

    /// Custom path score (combined scores over the lattice) for this arc
    double custom_path_score;

    SegmentedArc(int arc_id_, std::string label_, int tr_id_, int source_node_, int target_node_, double arc_score_, double acoustic_score_, double total_score_) : net_arc_id(arc_id_), label(label_), transition_index(tr_id_), source_node(source_node_), target_node(target_node_), arc_score(arc_score_), arc_acoustic_score(acoustic_score_), total_score(total_score_) { custom_score = 0; custom_path_score = 0; }
  };

  /** Segmented lattice node */
  struct SegmentedNode {
    int frame; //!< Entrance frame number (the first frame to be consumed)

    std::vector<int> in_arcs; //!< SegmentedArc ids
    std::vector<int> out_arcs; //!< SegmentedArc ids

    SegmentedNode() { frame = -1; }
    SegmentedNode(int frame_) : frame(frame_) { }
  };

  /** Segmented lattice containing the arcs and nodes */
  class SegmentedLattice {
  public:
    /** true if every arc consumes exactly one frame.
     * If false, the net_arc_id fields of SegmentedArc's refer to
     * m_logical_arcs instead of m_arcs
     */
    bool frame_lattice;

    std::vector<SegmentedNode> nodes; //!< Segmented nodes
    std::vector<SegmentedArc> arcs; //!< Segmented arcs
    int initial_node; //!< Index of the initial node
    int final_node; //!< Index of the final node

    double total_score; //!< Log sum of all the lattice path scores
    double total_custom_score; //!< Combined custom path score

    /** If frame_lattice == false, this vector contains the child arcs
     *  in the corresponding child lattice (usually a frame-level lattice).
     */
    std::vector< std::vector<int> > child_arcs;

  public:
    SegmentedLattice() { frame_lattice = false; initial_node = final_node = -1; total_score = loglikelihoods.zero(); total_custom_score = 0; }

    /** Recomputes the total scores of the segmented lattice arcs.
     * Due to pruning, the arc total scores may not be exact after
     * \ref create_segmented_lattice. Calling this fixes those scores.
     * The difference between the scores before and after fixing is
     * largest in the smaller (less probable) scores.
     * This method is also called internally within
     * \ref extract_segmented_lattice.
     */
    void compute_total_scores(void);


    /** Queries the custom scores for the segmented lattice arcs and
     * applies forward-backward algorithm to compute the lattice level
     * custom path scores.
     * \param combination_mode  Custom score combination mode
     *                          (CUSTOM_AVG [default], CUSTOM_SUM or
     *                           CUSTOM_MAX)
     */
    void compute_custom_path_scores(CustomScoreQuery *callback,
                                    int combination_mode=CUSTOM_AVG);

    /** Copies the custom path scores of a higher hierarchy segmented lattice
     * to a frame level segmented lattice
     * \param sl  Segmented lattice for which the arc custom scores
     *            have been filled with \ref fill_scutom_scores
     * \param frame_sl  Target segmented lattice, for which custom_path_score
     *                  fields are filled
     * \param combination_mode  Custom score combination mode
     *                          (CUSTOM_AVG [default], CUSTOM_SUM or
     *                          CUSTOM_MAX)
     */
    void propagate_custom_scores_to_frame_segmented_lattice(
      SegmentedLattice *frame_sl, int combination_mode=CUSTOM_AVG);

    /** Write the segmented lattice in FST format
     * \param file FILE handle opened for writing
     * \param arc_total_scores If set to true, the arc scores written to
     *                         the file are total scores. The default is
     *                         to write "instant" arc scores instead of
     *                         totals.
     */
    void write_segmented_lattice_fst(FILE *file, bool arc_total_scores = false);

    /** Save the segmented lattice in a custom format which can be loaded
     * later. Only for frame level lattices!
     * \param file FILE handle opened for writing
     */
    void save_segmented_lattice(FILE *file);

    /** Load a saved segmented lattice.
     * \param file FILE handle opened for writing
     */
    void load_segmented_lattice(FILE *file, HmmNetBaumWelch &parent);

    /** Recomputes the total custom score
     * \param combination_mode  Custom score combination mode
     *                          (CUSTOM_AVG [default], CUSTOM_SUM or
     *                           CUSTOM_MAX)
     */
    void recompute_custom_path_scores(int combination_mode=CUSTOM_AVG) { compute_custom_path_scores(NULL, combination_mode); }



  private:
    /** Create a new arc into this segmented lattice
     * \returns Index of the new arc in the arc vector
     */
    int create_segmented_arc(int arc_id, std::string &label,
                             int transition_index,
                             int source_seg_node, int target_seg_node,
                             double arc_score, double acoustic_score,
                             double total_score);
    
    /** Query the custom scores for all the arcs of the segmented lattice
     */
    void fill_custom_scores(CustomScoreQuery *callback);
    
    /// Combines custom scores with the given combination mode
    double combine_custom_scores(double log_score, double custom_score,
                                 double old_log_score,
                                 double old_custom_score,
                                 int combination_mode);

    friend class HmmNetBaumWelch;
  };


private:

  /******************* Private type definitions *******************/

  /** Class for handling the lattice labels while loading the lattices
   */
  class LatticeLabel {
  public:
    LatticeLabel();
    LatticeLabel(std::string in_str, std::string out_str);
    LatticeLabel(int tr_id, std::string raw_label);
    bool is_last(void) const { return last; }
    bool is_valid(void) const { return (label.size() > 0); }
    bool is_epsilon(void) const { return epsilon; }
    LatticeLabel higher_level_label(void) const;
    const std::string& get_label(void) const { return label; }
    int get_transition_index(void) const { return transition_index; }

  private:
    void initialize_labels(const std::string &raw_label);
    std::string remove_end_marks(const std::string &str) const;
    
  private:
    std::string original_label;
    std::string label;
    int transition_index;
    bool last; // Propagate to next node?
    bool epsilon;
  };

  /** Pending arcs are segmented arcs waiting to be realized
   */
  struct PendingArc {
    int arc_id;
    int source_seg_node;
    double arc_score;     // Cumulative arc scores
    double arc_acoustic_score; // Acoustic (and transition) score of arc_id
    double forward_score; // Includes the arc score(s)
    double total_score;
    
    PendingArc(int arc_id_, int source_node_, double arc_score_, double acoustic_score_, double forward_score_, double total_score_) : arc_id(arc_id_), source_seg_node(source_node_), arc_score(arc_score_), arc_acoustic_score(acoustic_score_), forward_score(forward_score_), total_score(total_score_) { }
  };


  struct BackwardToken {
    int node_id; //!< Network node id
    double score; //!< Backward score
    BackwardToken(int net_node_, double score_) : node_id(net_node_), score(score_) { }
  };

  struct BackwardTransitionInfo {
    int arc_id;
    double score;
    BackwardTransitionInfo(int arc_id_, double score_) : arc_id(arc_id_), score(score_) { }
  };

  typedef std::map< int, BackwardTransitionInfo > ParentArcTransitionMap;
  typedef std::multimap< int, BackwardTransitionInfo > NodeTransitionMap;
  
  /** Token information for forward pass */
  struct ForwardToken {
    int node_id; //!< Network node id
    double score; //!< Forward score
    std::set<int> pending_arcs; //!< Indices to a table of pending arcs
    int source_seg_node; //!< Source segmented node index for the next arc
    
    ForwardToken(int net_node_, double score_) : node_id(net_node_), score(score_) { source_seg_node = -1; }
  };

  typedef std::map< int, int > NodeTokenMap;

  // Internal type for storing log scores and custom path scores
  struct ScorePair {
    double score;
    double custom_score;
    ScorePair(double score_, double custom_) : score(score_), custom_score(custom_) { }
  };

  
public:
  /******************* Public interface *******************/
  
  HmmNetBaumWelch(FeatureGenerator &fea_gen, HmmSet &model);
  virtual ~HmmNetBaumWelch();

  /** Read the network from a file in mitfst ascii format. */
  void read_fst(FILE *file);

  /** Reads a segmented lattice from a file */
  SegmentedLattice* load_segmented_lattice(std::string &file);

  /** Generates features for the lattice objects. Note! Usually the
   * features are generated automatically via \ref create_segmented_lattice,
   * this function may be needed if loading a precomputed segmented lattice
   * and rescoring it
   */
  void generate_features(void);
  
  /** Set the pruning thresholds.
   * If the threshold is zero, the current value will not be changed.
   * \param forward       Forward beam (second pass)
   * \param backward      Backward beam (first pass)
   */
  void set_pruning_thresholds(double forward, double backward);

  double get_backward_beam(void) { return m_backward_beam; }
  double get_forward_beam(void) { return m_forward_beam; }

  /// Set the segmentation mode
  void set_mode(int mode) { m_segmentation_mode = mode; }

  /// Set the scaling for acoustic log likelihoods
  void set_acoustic_scaling(double scale) { m_acoustic_scale = scale; }

  /// Set the use of search network static scores
  void set_use_static_scores(bool use) { m_use_static_scores = use; }

  /// Set the use of transition_probabilities
  void set_use_transition_probabilities(bool use) { m_use_transition_probabilities = use; }

  /** Runs forward-backward algorithm and creates a segmented lattice
      representation. Note that due to pruning, the arc total scores
      may not be exact before calling
      \ref compute_total_scores of the SegmentedLattice.
  */
  SegmentedLattice* create_segmented_lattice(void);
  
  /** Extracts a higher hierarchy lattice from a frame-level segmented
      lattice.
  */
  SegmentedLattice* extract_segmented_lattice(SegmentedLattice *frame_sl,
                                              int level);

  /** Recomputes the acoustic scores of a frame segmented lattice,
   * updates the arc scores and finally computes new total score.
   * \param frame_sl Pointer to frame lattice
   */
  void rescore_segmented_lattice(SegmentedLattice *frame_sl);



  /** Return the feature vector for a frame */
  const FeatureVec get_feature(int frame) const { assert( m_features_generated ); return m_features[frame]; }
  

  // Segmentator interface
  virtual void open(std::string ref_file);
  virtual void close();
  virtual void set_frame_limits(int first_frame, int last_frame);
  virtual void set_collect_transition_probs(bool collect) { m_collect_transitions = collect; }
  virtual bool init_utterance_segmentation(void);
  virtual int current_frame(void) { return m_cur_frame; }
  virtual bool next_frame(void);
  virtual void reset(void);
  virtual bool eof(void) { return m_eof_flag; }
  virtual bool computes_total_log_likelihood(void) { return true; }
  virtual double get_total_log_likelihood(void) { return (m_segmentator_seglat == NULL ? loglikelihoods.zero() : m_segmentator_seglat->total_score); }
  virtual const Segmentator::IndexProbMap& pdf_probs(void) { return m_pdf_prob_map; }
  virtual const Segmentator::IndexProbMap& transition_probs(void) { return m_transition_prob_map; }
  virtual const std::string& highest_prob_label(void) { return m_most_probable_label; }


private:
  /******************* Private methods *******************/

  /** Checks the network structure and fills the epsilon flags for nodes.
   * Throws an exception if invalid structure is detected.
   */
  void check_network_structure(void);

  /** Fixes the parent arc id of an incoming branch if a parent arc
   * with the same label already exists.
   * \param arc_id New incoming arc
   * \returns Hierarchy level at which the replace took place, or -1
   */
  int fix_parent_arcs(int arc_id);

  /** Replaces the given parent arc with the new parent arc in the specified
   * hierarchy level. Propagates the change to the other arcs entering or
   * leaving the source/target node (depending on the direction).
   */
  int replace_branch_parent_arc(int arc_id, int parent_level,
                                int new_parent_id, bool forward,
                                std::set<int> &processed_arcs);

  bool fill_backward_probabilities(void);
  void generate_segmented_lattice(void);

  /** Propagates the epsilon arcs of the active nodes
   */
  void backward_propagate_epsilon_arcs(
    std::vector< BackwardToken > &active_tokens,
    NodeTokenMap &node_token_map, int cur_frame);
  
  /** Returns the arc score */
  double get_arc_score(int arc_id, const FeatureVec &fea_vec);

  /** Either creates or updates a token (if a token already exists) */
  ForwardToken& create_or_update_token(
    std::vector< ForwardToken > &token_vector, NodeTokenMap &node_token_map,
    int node_id, double forward_score);

  int esl_merge_child_arcs(int leaf1, int leaf2,
                           std::vector< std::pair<int,int> > &tree);
  void esl_fill_child_arcs(std::vector<int> &child_arcs, int leaf_index,
                           std::vector< std::pair<int,int> > &tree);

  /** Clears the backward-phase scores */
  void clear_bw_scores(void);

  
private:
  /******************* Private data *******************/
  
  FeatureGenerator &m_fea_gen;
  HmmSet &m_model;

  std::string m_epsilon_string;

  /// Cache of generated features
  FeatureBuffer m_features;
  bool m_features_generated;

  /* Search network */
  int m_initial_node_id; //!< Index of the unique initial node
  int m_final_node_id; //!< Index of the unique final node
  std::vector<Node> m_nodes; //!< Nodes of the search network
  std::vector<Arc> m_arcs; //!< Arcs of the search network

  /// Logical arcs of the search network
  std::vector<LogicalArc> m_logical_arcs;

  /// Segmentation mode
  int m_segmentation_mode;

  /// Whether to include search network static scores to path scores
  bool m_use_static_scores;

  /// Whether to include transition probabilities to path scores
  bool m_use_transition_probabilities;
  
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

  /** Maximum score or sum of path scores, depending on the segmentation mode.
   * Computed in the backward phase.
   */
  double m_total_score;

  /// True if backward scores have been computed
  bool m_bw_scores_computed;


  // For segmentation interface
  SegmentedLattice *m_segmentator_seglat;
  int m_cur_frame; //!< Current frame
  std::set<int> m_active_nodes; //! Current active nodes
  /// A vector of possible PDFs and their probabilities
  Segmentator::IndexProbMap m_pdf_prob_map;
  /// A vector of possible transitions and their probabilities
  Segmentator::IndexProbMap m_transition_prob_map;
  /// String containing the current most probable arc label
  std::string m_most_probable_label;
  
  /// true if eof has been detected or frame limits have been reached
  bool m_eof_flag;

  /// true if transitions are to be collected
  bool m_collect_transitions;

  friend class SegmentedLattice;

};

}

#endif // HMMNETBAUMWELCH_HH
