#ifndef LM_HH
#define LM_HH

#include <float.h>
#include "misc/types.hh"
#include "FixedArray.hh"

namespace fsalm {

  class Semiring {
  public:
    Semiring() : one(0), zero(-1e30) { }
    virtual ~Semiring() { }
    virtual float plus(float a, float b) = 0;
    virtual float times(float a, float b) = 0;
    virtual float divide(float a, float b) = 0;
    virtual bool equal(float a, float b) = 0;
    float one;
    float zero;
  };

  class MaxPlusSemiring : public Semiring {
  public:
    virtual ~MaxPlusSemiring() { }
    virtual float plus(float a, float b) { return a < b ? b : a; }
    virtual float times(float a, float b) { return a + b; }
    virtual float divide(float a, float b) { return a - b; }
    virtual bool equal(float a, float b) { return a == b; }
  };

  /** Class for storing ngram language model in a fst format with
   * backoff arcs.  
   *
   * INVARIANTS:
   *
   * - before trim() there may be nodes without children and bt=0
   * 
   * - after trim() only final node has bt=0 and has zero children
   *
   * - node n has zero children if limit[n-1] == limit[n] or limit[n] == 0
   *
   * IMPLEMENTATION NOTES:
   *
   * - Backoff information is stored in the nodes instead of normal
   * arcs.  Otherwise, creating the arcs in proper order would be
   * troublesome from ARPA format.
   *
   * ASSUMES:
   * - children of a context are inserted in sorted chunk
   * - lower order is inserted before higher order
   */
  class LM {
  public:

    typedef FixedArray<int> Array;
    typedef FixedArray<float> FloatArray;

    Str start_str;
    Str end_str;
    Semiring *semiring;

    LM();
    void reset();
    const SymbolMap &symbol_map() const { return m_symbol_map; }
    int start_symbol() const { return m_start_symbol; }
    int end_symbol() const { return m_end_symbol; }
    int final_node_id() const { return m_final_node_id; }
    int empty_node_id() const { return m_empty_node_id; }
    int initial_node_id() const { return m_initial_node_id; }
    int order() const { return m_order; }
    int num_arcs() const { return m_arcs.symbol.num_elems(); }
    int num_nodes() const { return m_nodes.bo_target.num_elems(); }
    float final_score() const { return m_final_score; }

    /** Return the number of explicit children of node. */
    int num_children(int node_id) const;

    /** String describing a vector of symbol indices. */
    Str str(const IntVec &vec) const;

    /** Find context node for given backoff vector by backoffing
     * further if necessary.  
     *
     * \note Throws exception if given ngram ends in sentence end
     * symbol.
     *
     * \param vec = candidate context for the backoff
     * \return node index corresponding to backoff context 
     */
    int find_backoff(IntVec vec) const;

    /** Move to next node throught given symbol but do not backoff.
     *
     * \param node_id = the node to start from (final node not allowed)
     * \param symbol = the symbol to search
     * \param score = float pointer to which the possible score is ADDED 
     * \return the resulting node (or -1 if explicit arc with symbol not found)
     */
    int walk_no_bo(int node_id, int symbol, float *score = NULL) const;
    int walk_no_bo(int node_id, const IntVec &vec, float *score = NULL) const;

    /** Move to next node through given symbols without backoffing.
     * Same as above, but returns node indices for each symbol in a
     * vector. */
    IntVec walk_no_bo_vec(int node_id, const IntVec &vec, 
                          float *score = NULL) const;

    /** Move to next node through given symbol by backoffing if necessary.
     *
     * \param node_id = the node to start from (final node not allowed)
     * \param symbol = the symbol to traverse through
     * \param score = float pointer to which the score is ADDED
     * \return the resulting node
     */
    int walk(int node_id, int symbol, float *score = NULL) const;
    int walk(int node_id, const IntVec &vec, float *score = NULL) const;

    void new_arc(int src_node_id, int symbol, int tgt_node_id, float score);
    void new_ngram(const IntVec &vec, float score, float bo_score);
    int new_node();
    void set_arc(int arc_id, int symbol, int target, float score);
    void trim();
    void quantize(int bits);

    /** Compute potential of each node (used by push). */
    void compute_potential(FloatVec &d);

    /** Push scores as early as possible. */
    void push();

    void read_arpa(FILE *file, bool show_progress = false);
    void read(FILE *file);
    void write(FILE *file) const;

    /** Write the language model in MIT fst format. */
    void write_fst(FILE *file, Str bo_symbol = "<B>") const;

    /** Write the language model in ATT fsmt format. */
    void write_fsmt(FILE *file, Str bo_symbol = "<B>") const;
    void write_fsmt_node(FILE *file, int n, Str bo_symbol) const;

    /** Fetch probabilities for all symbols (ignoring non-events). */
    void fetch_probs(int node_id, FloatVec &vec);

    Str debug_str() const;
    bool debug_check_sum(int node_id) const;
    bool debug_check_sums() const;
    bool debug_check_zero_bo() const;

  private:

    /** Incoming arc used temporarily by compute_potential() */
    struct InArc {
      InArc() : source(-1), arc_id(-1) { }
      InArc(int source, int arc_id) : source(source), arc_id(arc_id) { }
      int source;
      int arc_id;
    };

    /** Node information */
    struct {
      FloatArray bo_score;
      Array bo_target;
      Array limit_arc;
    } m_nodes;

    /** Arc information */
    struct {
      Array symbol;
      Array target;
      FloatArray score;
    } m_arcs;

    /** Cache containing information about the last ngram inserted in
     * the strucure. */
    struct {
      IntVec ctx_vec;
      int ctx_node_id;
    } m_cache;

    /** Mapping between language model symbols and indices. */
    SymbolMap m_symbol_map;

    /** Bit array defining non-event symbols. */
    BoolVec m_non_event;

    int m_order;
    int m_empty_node_id;
    int m_initial_node_id;
    int m_final_node_id;
    int m_start_symbol;
    int m_end_symbol;
    float m_final_score;
  };

};

#endif /* LM_HH */
