#ifndef LM_HH
#define LM_HH

#include <cstddef>
#include <cfloat>
#include <string>
#include <vector>

#include "misc/SymbolMap.hh"


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
    virtual float plus(float a, float b) {
        return a < b ? b : a;
    }
    virtual float times(float a, float b) {
        return a + b;
    }
    virtual float divide(float a, float b) {
        return a - b;
    }
    virtual bool equal(float a, float b) {
        return a == b;
    }
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

    std::string start_str;
    std::string end_str;
    Semiring *semiring;

    LM();
    void reset();
    const misc::SymbolMap<std::string,int> &symbol_map() const {
        return m_symbol_map;
    }
    int start_symbol() const {
        return m_start_symbol;
    }
    int end_symbol() const {
        return m_end_symbol;
    }
    int final_node_id() const {
        return m_final_node_id;
    }
    int empty_node_id() const {
        return m_empty_node_id;
    }
    int initial_node_id() const {
        return m_initial_node_id;
    }
    int order() const {
        return m_order;
    }
    int num_arcs() const {
        return m_arcs.symbol.size();
    }
    int num_nodes() const {
        return m_nodes.bo_target.size();
    }
    float final_score() const {
        return m_final_score;
    }

    /** Return the number of explicit children of node. */
    int num_children(int node_id) const;

    /** String describing a vector of symbol indices. */
    std::string str(const std::vector<int> &vec) const;

    /** Find context node for given backoff vector by backoffing
     * further if necessary.
     *
     * If possible, traverses the sequence of symbols in \a vec, and returns the
     * resulting node. Otherwise removes the first symbol until such a sequence
     * is found.
     *
     * \note Throws exception if given ngram ends in sentence end
     * symbol.
     *
     * \param vec = candidate context for the backoff
     * \return node index corresponding to backoff context
     */
    int find_backoff(std::vector<int> vec) const;

    /** Move to next node through given symbol but do not backoff.
     *
     * Finds the arc that starts from given node and has the given
     * symbol assigned, then returns its target node.
     *
     * \param node_id The node to start from (final node not allowed), or
     * empty_node_id() to search every node(?)
     * \param symbol The symbol to search for.
     * \param score Float pointer to which the possible score of the arc is ADDED
     * \return The resulting node (or -1 if explicit arc with symbol not found).
     */
    int walk_no_bo(int node_id, int symbol, float *score = NULL) const;

    /** Walks through a list of symbols but does not backoff. Returns the index
     * of the resulting node.
     *
     * Finds a path that starts from given node and has the list of symbols
     * assigned in given order, then returns the target node of the last arc.
     *
     * \param node_id The node to start from (final node not allowed), or
     * empty_node_id() to search every node(?)
     * \param vec The list of symbols to traverse.
     * \param score Float pointer to which the possible score of the arc is ADDED
     * \return The resulting node (or -1 if explicit arc with symbol not found).
     */
    int walk_no_bo(int node_id, const std::vector<int> &vec, float *score = NULL) const;

    /** Walks through a list of symbols but does not backoff. Returns indices of
     * all the traversed nodes.
     *
     * Finds a path that starts from given node and has the list of symbols
     * assigned in given order, then returns the target node of the last arc.
     *
     * \param node_id The node to start from (final node not allowed), or
     * empty_node_id() to search every node(?)
     * \param vec The list of symbols to traverse.
     * \param score Float pointer to which the possible score of the arc is ADDED
     * \return Indices of all the traversed nodes.
     */
    std::vector<int> walk_no_bo_vec(int node_id, const std::vector<int> &vec,
                                    float *score = NULL) const;

    /** Move to next node through given symbol by backoffing if necessary.
     *
     * \param node_id = the node to start from (final node not allowed)
     * \param symbol = the symbol to traverse through
     * \param score = float pointer to which the score is ADDED
     * \return the resulting node
     */
    int walk(int node_id, int symbol, float *score = NULL) const;
    int walk(int node_id, const std::vector<int> &vec, float *score = NULL) const;

    void new_arc(int src_node_id, int symbol, int tgt_node_id, float score);
    void new_ngram(const std::vector<int> &vec, float score, float bo_score);
    int new_node();
    void set_arc(int arc_id, int symbol, int target, float score);
    void trim();
    void quantize(int bits);

    /** Compute potential of each node (used by push). */
    void compute_potential(std::vector<float> &d);

    /** Push scores as early as possible. */
    void push();

    /** Reads the language model in ARPA format. */
    void read_arpa(FILE *file, bool show_progress = false);
    /** Reads the language model in a non-standard format. */
    void read(FILE *file);
    /** Writes the language model in a non-standard format. */
    void write(FILE *file) const;

    /** Write the language model in MIT fst format. */
    void write_fst(FILE *file, std::string bo_symbol = "<B>") const;

    /** Write the language model in ATT fsmt format. */
    void write_fsmt(FILE *file, std::string bo_symbol = "<B>") const;
    void write_fsmt_node(FILE *file, int n, std::string bo_symbol) const;

    /** Fetch probabilities for all symbols (ignoring non-events). */
    void fetch_probs(int node_id, std::vector<float> &vec);

    std::string debug_str() const;
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
        std::vector<float> bo_score;
        std::vector<int> bo_target;
        std::vector<int> limit_arc;  //!< Index to one past the last arc that starts from this node.
    } m_nodes;

    /** Arc information */
    struct {
        std::vector<int> symbol;  //!< The symbol assigned to the arc.
        std::vector<int> target;  //!< The target node.
        std::vector<float> score;  //!< Possible score of the arc.
    } m_arcs;

    /** Cache containing information about the last ngram inserted in
     * the strucure. */
    struct {
        std::vector<int> ctx_vec;  //!< Context vector, i.e. all but the last symbol.
        int ctx_node_id;
    } m_cache;

    /** Mapping between language model symbols and indices. */
    misc::SymbolMap<std::string,int> m_symbol_map;

    /** Bit array defining non-event symbols. */
    std::vector<bool> m_non_event;

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
