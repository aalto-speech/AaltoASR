#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <iostream>

#include "LM.hh"
#include "ArpaReader.hh"

using namespace std;


namespace fsalm {

static MaxPlusSemiring maxplus_semiring;

float log10_to_ln(float f)
{
    return f * M_LN10;
}

template <class myVec>
void vec_resize(myVec &vec, int sz) {
    if (sz > vec.size()) {
        int cap = vec.capacity();
        if (sz >= cap) {
            int new_cap = cap * 2;
            if (sz >= new_cap)
                new_cap = sz + 1;
            vec.reserve(new_cap);
        }
        vec.resize(sz);
    }
}


/** Write the vector to file.
 *
 * \param file = file stream to write to
 * \throw runtime_error if write fails
 */
template <class myVec>
void vec_write(myVec &vec, FILE *file)
{
    fputs("LMVECTOR1:", file);
    fprintf(file, "%d:", (int)vec.size());
    int data_len = vec.size() * sizeof(vec[0]);
    if (data_len > 0) {
        size_t ret = fwrite((unsigned char*)&vec.at(0), data_len, 1, file);
        if (ret != 1)
            throw runtime_error(
                string("vec_write() fwrite failed: ") +
                strerror(errno));
    }
}


/** Read the vector from file.
 *
 * \param file = file stream to read from
 * \throw runtime_error if read fails
 */
template <class myVec>
void vec_read(myVec &vec, FILE *file)
{
    int version;
    int num_elems;
    int ret = fscanf(file, "LMVECTOR%d:%d:",
                     &version, &num_elems);
    if (ret != 2 || version != 1)
        throw runtime_error("vec_read() error while reading header");
    vec.resize(num_elems);
    int data_len = vec.size() * sizeof(vec[0]);
    if (data_len > 0) {
        size_t ret = fread((unsigned char*)&vec.at(0), data_len, 1, file);
        if (ret != 1) {
            if (ferror(file))
                throw runtime_error("vec_read() error while reading buffer");
            assert(feof(file));
            throw runtime_error("vec_read() eof while reading buffer");
        }
    }
}


LM::LM()
{
    start_str = "<s>";
    end_str = "</s>";
    semiring = &maxplus_semiring;
    reset();
}

void LM::reset()
{
    m_order = 0;
    m_final_node_id = -1;
    m_empty_node_id = -1;
    m_initial_node_id = -1;
    m_start_symbol = -1;
    m_end_symbol = -1;
    m_final_score = 0;

    m_symbol_map.clear();
    m_non_event.clear();
    m_nodes.bo_score.clear();
    m_nodes.bo_target.clear();
    m_nodes.limit_arc.clear();
    m_arcs.symbol.clear();
    m_arcs.target.clear();
    m_arcs.score.clear();
    m_cache.ctx_vec.clear();
    m_cache.ctx_node_id = -1;
}

int LM::num_children(int node_id) const
{
    assert(node_id >= 0);
    if (node_id == 0)
        return 0;
    int limit = m_nodes.limit_arc.at(node_id);
    if (limit == 0)
        return 0;
    int first = m_nodes.limit_arc.at(node_id - 1);
    assert(limit >= first);
    return limit - first;
}

string LM::str(const vector<int> &vec) const
{
    string ret;
    for (int i=0; i<vec.size(); i++) {
        if (i > 0)
            ret.append(" ");
        ret.append(m_symbol_map.at(vec[i]));
    }
    return ret;
}

int LM::find_backoff(vector<int> vec) const
{
    if (vec.empty())
        return m_empty_node_id;
    if (vec.back() == m_end_symbol)
        throw runtime_error("LM::find_backoff(): sentence end not allowed");

    while (1) {
        vector<int> nodes = walk_no_bo_vec(m_empty_node_id, vec);
        assert(!nodes.empty());
        assert(nodes.size() <= vec.size());
        if (nodes.size() == vec.size())
            return nodes.back();
        vec.erase(vec.begin());
        assert(!vec.empty());
    }
}

int LM::walk_no_bo(int node_id, int symbol, float *score) const
{
    assert(node_id >= 0);
    if (node_id == m_final_node_id)
        throw runtime_error("LM::walk_no_bo(): final node not allowed");
    // Limit tells the first arc that will not be considered in the search.
    int limit = m_nodes.limit_arc.at(node_id);
    if (limit > 0) {
        // The first arc that will be considered in the search is limit_arc of the previous node.
        int first = m_nodes.limit_arc.at(node_id - 1);
        assert(limit >= first);
        if (limit > first) {
            // Find an arc with the given symbol.
            auto lower_b=lower_bound (m_arcs.symbol.begin()+first, m_arcs.symbol.begin()+limit, symbol);
            int arc_id = lower_b-m_arcs.symbol.begin();
            if (arc_id != limit && *lower_b == symbol) {
                // The search found such an arc.
                if (score != NULL)
                    *score += m_arcs.score.at(arc_id);
                return m_arcs.target.at(arc_id);
            }
        }
    }
    return -1;
}

int LM::walk_no_bo(int node_id, const vector<int> &vec, float *score) const
{
    for (int i=0; i<vec.size(); i++) {
        node_id = walk_no_bo(node_id, vec[i], score);
        if (node_id < 0)
            return -1;
    }
    return node_id;
}

vector<int> LM::walk_no_bo_vec(int node_id, const vector<int> &vec, float *score) const
{
    vector<int> ret;
    ret.reserve(vec.size());
    for (int i=0; i<vec.size(); i++) {
        node_id = walk_no_bo(node_id, vec[i], score);
        if (node_id < 0)
            break;
        ret.push_back(node_id);
    }
    return ret;
}

int LM::walk(int node_id, int symbol, float *score) const
{
    assert(symbol >= 0);
    while (1) {
        int new_node_id = walk_no_bo(node_id, symbol, score);
        if (new_node_id < 0) {
            if (score != NULL)
                *score += m_nodes.bo_score.at(node_id);
            node_id = m_nodes.bo_target.at(node_id);
            continue;
        }
        return new_node_id;
    }
}

int LM::walk(int node_id, const vector<int> &vec, float *score) const
{
    for (int i=0; i<vec.size(); i++) {
        node_id = walk(node_id, vec[i], score);
    }
    return node_id;
}

void LM::new_arc(int src_node_id, int symbol, int tgt_node_id, float score)
{
    assert(src_node_id > 0);
    int arc_id = num_arcs();
    int limit_arc = m_nodes.limit_arc.at(src_node_id);
    assert(limit_arc == 0 || limit_arc == arc_id);

    // Set limit_arc for nodes that precede src_node_id and have
    // limit_arc unset.
    if (limit_arc == 0 && arc_id > 0) {
        for (int n = src_node_id - 1; n > 0; n--) {
            if (m_nodes.limit_arc.at(n) > 0)
                break;
            vec_resize(m_nodes.limit_arc, n + 1);
            m_nodes.limit_arc.at(n) = arc_id;
        }
    }

    vec_resize(m_nodes.limit_arc, src_node_id + 1);
    m_nodes.limit_arc.at(src_node_id) = arc_id + 1;
    set_arc(arc_id, symbol, tgt_node_id, score);
}

void LM::new_ngram(const vector<int> &vec, float score, float bo_score)
{
    assert(!vec.empty());

    // If the context is not the same as on the previous insertion, find the
    // context node by traversing the arcs specified by the context symbols
    // (all but the last symbol in the n-gram).
    //
    vector<int> ctx_vec(vec.begin(), vec.end() - 1);
    if (m_cache.ctx_node_id < 0 || ctx_vec != m_cache.ctx_vec) {
        m_cache.ctx_node_id = walk_no_bo(empty_node_id(), ctx_vec);
        if (m_cache.ctx_node_id < 0)
            throw runtime_error("prefix missing for ngram \"" + str(vec) + "\"");
        m_cache.ctx_vec = ctx_vec;
    }
    assert(m_cache.ctx_node_id > 0);


    // Find the target and backoff node ids, and possibly create the
    // new node.
    //
    int bo_node_id = -1;
    int tgt_node_id = -1;
    if (vec.back() == m_end_symbol) {
        tgt_node_id = m_final_node_id;
        bo_node_id = m_final_node_id;
    }
    else {
        bo_node_id = find_backoff(vector<int>(vec.begin() + 1, vec.end()));
        assert(bo_node_id >= 0);

        // For highest order n-grams, an arc to the backoff node will be created.
        // For other n-grams, a new target node is created.
        if (vec.size() == order()) {
            tgt_node_id = bo_node_id;
            assert(bo_score == 0);
        }
        else
            tgt_node_id = new_node();
    }
    if (vec.size() == 1 && vec.back() == m_start_symbol)
        m_initial_node_id = tgt_node_id;

    // Create arc from the context node to the target node, and update possible
    // backoff information.
    //
    new_arc(m_cache.ctx_node_id, vec.back(), tgt_node_id, score);
    if (tgt_node_id != bo_node_id) {
        vec_resize(m_nodes.bo_target, tgt_node_id + 1);
        m_nodes.bo_target.at(tgt_node_id) = bo_node_id;
        vec_resize(m_nodes.bo_score, tgt_node_id + 1);
        m_nodes.bo_score.at(tgt_node_id) = bo_score;
    }
}

int LM::new_node()
{
    int node_id = num_nodes();
    vec_resize(m_nodes.bo_target, node_id + 1);
    m_nodes.bo_target.at(node_id) = 0;
    vec_resize(m_nodes.bo_score, node_id + 1);
    m_nodes.bo_score.at(node_id) = 0;
    vec_resize(m_nodes.limit_arc, node_id + 1);
    m_nodes.limit_arc.at(node_id) = 0;
    return node_id;
}

void LM::set_arc(int arc_id, int symbol, int target, float score)
{
    vec_resize(m_arcs.symbol, arc_id + 1);
    m_arcs.symbol.at(arc_id) = symbol;
    vec_resize(m_arcs.target, arc_id + 1);
    m_arcs.target.at(arc_id) = target;
    vec_resize(m_arcs.score, arc_id + 1);
    m_arcs.score.at(arc_id) = score;
}

void LM::trim()
{
    // Find childless nodes and compute new node indices by not
    // counting childless nodes and backoffing for removed nodes.
    //
    vector<int> new_target(num_nodes(), 0);
    vector<bool> removed(num_nodes(), false);
    new_target.at(0) = 0;
    int new_n = 1;
    for (int n = 1; n < num_nodes(); n++) {
        if (num_children(n) == 0) {
            float bo_score = m_nodes.bo_score.at(n);
            if (bo_score != 0)
                fprintf(stderr, "WARNING: LM::trim(): childless node %d "
                        "with bo_score = %g\n", n, bo_score);

//          throw runtime_error(str::fmt(256, "LM::trim(): childless node %d "
//                               "with bo_score = %g", n, bo_score));
            new_target[n] = new_target.at(m_nodes.bo_target.at(n));
            removed[n] = true;
        }
        else
            new_target[n] = new_n++;
    }

    // Correct arc target indices.  Replace childless targets by
    // backoff targets.
    //
    for (int a = 0; a < num_arcs(); a++) {
        int tgt = m_arcs.target.at(a);
        m_arcs.target.at(a) = new_target.at(tgt);
    }

    // Remove childless nodes and update backoff target indices.
    //
    for (int n = 1; n < num_nodes(); n++) {
        if (removed[n])
            continue;
        m_nodes.bo_score.at(new_target[n]) = m_nodes.bo_score.at(n);
        m_nodes.bo_target.at(new_target[n]) = new_target.at(m_nodes.bo_target.at(n));
        m_nodes.limit_arc.at(new_target[n]) = m_nodes.limit_arc.at(n);
    }
    vec_resize(m_nodes.bo_score, new_n);
    vec_resize(m_nodes.bo_target, new_n);
    vec_resize(m_nodes.limit_arc, new_n);

    // Update initial node id
    m_initial_node_id = walk(m_empty_node_id, m_start_symbol);
}

/** Apply linear quantization to log probability values.
 *
 * \param bits = number of bits to use for floats (remember that
 * sign occupies on bit from the given number)
 */
void LM::quantize(int bits)
{
    assert(false);
//    m_nodes.bo_score.linear_quantization_bits(bits);
//    m_arcs.score.linear_quantization_bits(bits);
}

void LM::compute_potential(vector<float> &d)
{
    // Implemented according to "Mehryar Mohri and Michael Riley.  A
    // Weight Pushing Algorithm for Large Vocabulary Speech
    // Recognition.  Eurospeech 2001."

    // Compute incoming arcs for each node.
    vector<vector<InArc> > in_arcs;
    in_arcs.resize(num_nodes());
    for (int n = 0; n < num_nodes(); n++) {
        int bo_tgt = m_nodes.bo_target.at(n);
        if (bo_tgt > 0)
            in_arcs.at(bo_tgt).push_back(InArc(n, -1));

        int limit = m_nodes.limit_arc.at(n);
        if (limit == 0)
            continue;
        int first = m_nodes.limit_arc.at(n-1);
        assert(first <= limit);
        for (int a = first; a < limit; a++) {
            int tgt = m_arcs.target.at(a);
            in_arcs.at(tgt).push_back(InArc(n, a));
        }
    }

    int N = num_nodes();
    d.clear();
    d.resize(N, semiring->zero);
    vector<float> r(N, semiring->zero);
    vector<bool> in_queue(N, false);
    deque<int> queue;

    d.at(m_final_node_id) = semiring->one;
    r.at(m_final_node_id) = semiring->one;
    in_queue.at(m_final_node_id) = true;
    queue.push_back(m_final_node_id);

    // Compute potentials in 'd'
    while (!queue.empty()) {
        int q = queue.front();
        queue.pop_front();
        assert(in_queue.at(q));
        in_queue.at(q) = false;

        float R = r.at(q);
        r.at(q) = semiring->zero;

        vector<InArc> &in_arcs_q = in_arcs.at(q);
        for (int a=0; a<in_arcs_q.size(); a++) {
            InArc &in_arc = in_arcs_q[a];
            int n = in_arc.source;

            float score = 0;
            if (in_arc.arc_id < 0)
                score = m_nodes.bo_score.at(n);
            else
                score = m_arcs.score.at(in_arc.arc_id);

            float R_times_a = semiring->times(R, score);
            float new_d = semiring->plus(d[n], R_times_a);
            if (!semiring->equal(d.at(n), new_d)) {
                d[n] = new_d;
                r[n] = semiring->plus(r.at(n), R_times_a);
                if (!in_queue.at(n)) {
                    in_queue[n] = true;
                    queue.push_back(n);
                }
            }
        }
    }
}

void LM::push()
{
    vector<float> potential;
    compute_potential(potential);

    assert(potential.size() == num_nodes());
    assert(potential.at(m_final_node_id) == semiring->one);

    m_final_score = potential.at(m_initial_node_id);

    for (int n = 0; n < num_nodes(); n++) {
        int bo_tgt = m_nodes.bo_target.at(n);
        if (bo_tgt > 0) {
            float score = m_nodes.bo_score.at(n);
            score = semiring->times(score, potential.at(bo_tgt));
            score = semiring->divide(score, potential.at(n));
            m_nodes.bo_score.at(n) = score;
        }

        int limit = m_nodes.limit_arc.at(n);
        if (limit == 0)
            continue;
        int first = m_nodes.limit_arc.at(n-1);
        assert(first <= limit);
        for (int a = first; a < limit; a++) {
            float score = m_arcs.score.at(a);
            int target = m_arcs.target.at(a);
            score = semiring->times(score, potential.at(target));
            score = semiring->divide(score, potential.at(n));
            m_arcs.score.at(a) = score;
        }

    }
}

void LM::read_arpa(FILE *file, bool show_progress)
{
    reset();

    m_final_node_id = new_node();
    m_empty_node_id = new_node();

    ArpaReader reader(file);
    reader.opt.show_progress = show_progress;
    reader.symbol_map = &m_symbol_map;
    reader.read_header();
    m_order = reader.header.order;

    // We have to read all 1-grams first to find out sentence start and
    // end symbols.
    //
    try {
        reader.read_order_ngrams(false);
    }
    catch (exception &e) {
        throw runtime_error(string("LM::read_arpa(): error while reading unigrams: ")
                            + e.what());
    }

    try {
        m_start_symbol = m_symbol_map.index(start_str);
        m_end_symbol = m_symbol_map.index(end_str);
    }
    catch (string &str) {
        throw runtime_error(
            "LM::read_arpa(): sentence start '" + start_str +
            "' or sentence end '" + end_str + "' not in unigrams");
    }

    // Call new_ngram() on each of the read 1-grams.
    //
    for (size_t i = 0; i < reader.order_ngrams.size(); i++) {
        ArpaReader::Ngram &ngram = reader.order_ngrams[i];
        assert(ngram.symbols.size() == 1);
        new_ngram(ngram.symbols, ngram.log_prob, ngram.backoff);
    }

    // Read and iterate through 2-grams, 3-grams, etc. in sorted order.
    // Insert using new_ngram() unless the first symbol is the start
    // symbol, or the last symbol is the end symbol.
    //
    while (reader.read_order_ngrams(true)) {
        for (int i = 0; i < (int)reader.sorted_order.size(); i++) {
            ArpaReader::Ngram &ngram = reader.order_ngrams[reader.sorted_order[i]];
            bool insert = true;
            for (size_t i = 0; i < ngram.symbols.size(); i++) {
                if ((ngram.symbols[i] == m_end_symbol && i+1 != ngram.symbols.size())
                        || (ngram.symbols[i] == m_start_symbol && i != 0))
                {
                    fprintf(stderr, "WARNING: skipping ngram '%s'\n",
                            str(ngram.symbols).c_str());
                    insert = false;
                    break;
                }
            }
            if (insert)
                new_ngram(ngram.symbols, ngram.log_prob, ngram.backoff);
        }
    }
    if (reader.num_ignored() > 0)
        fprintf(stderr, "WARNING: ignored %d ngrams in total\n",
                reader.num_ignored());

    m_non_event.clear();
    m_non_event.resize(m_symbol_map.size(), false);
    m_non_event.at(m_start_symbol) = true;

    fprintf(stderr, "fsalm: %d nodes, %d arcs\n", num_nodes(), num_arcs());
}


void LM::read(FILE *file)
{
    int version;
    int ret = fscanf(file, "LM%d:%d:%d:%d:%d:%g:",
                     &version, &m_order, &m_empty_node_id, &m_initial_node_id,
                     &m_final_node_id, &m_final_score);
    if (ret != 6 || version != 1)
        throw runtime_error("LM::read() error while reading header");
    str::read_line(start_str, file, true);
    str::read_line(end_str, file, true);
    m_symbol_map.read(file);
    vec_read(m_arcs.symbol, file);
    vec_read(m_arcs.target, file);
    vec_read(m_arcs.score, file);
    vec_read(m_nodes.bo_score, file);
    vec_read(m_nodes.bo_target, file);
    vec_read(m_nodes.limit_arc, file);

    m_start_symbol = m_symbol_map.index(start_str);
    m_end_symbol = m_symbol_map.index(end_str);

    m_non_event.clear();
    m_non_event.resize(m_symbol_map.size(), false);
    m_non_event.at(m_start_symbol) = true;

    // Check that initial node matches start symbol
    int node_id = walk(m_empty_node_id, m_start_symbol);
    if (node_id != m_initial_node_id)
        throw runtime_error(str::fmt(1024, "LM::read(): initial node %d does not match "
                                     "start symbol target %d", m_initial_node_id,
                                     node_id));
}

void LM::write(FILE *file) const
{
    fprintf(file, "LM1:%d:%d:%d:%d:%g:",
            m_order, m_empty_node_id, m_initial_node_id, m_final_node_id, m_final_score);
    fprintf(file, "%s\n%s\n", start_str.c_str(), end_str.c_str());
    m_symbol_map.write(file);
    vec_write(m_arcs.symbol, file);
    vec_write(m_arcs.target, file);
    vec_write(m_arcs.score, file);
    vec_write(m_nodes.bo_score, file);
    vec_write(m_nodes.bo_target, file);
    vec_write(m_nodes.limit_arc, file);
}


void LM::write_fst(FILE *file, string bo_symbol) const
{
    fputs("#FSTBasic MaxPlus\n", file);
    fprintf(file, "I %d\n", m_initial_node_id);
    fprintf(file, "F %d\n", m_final_node_id);

    for (int n = 1; n < num_nodes(); n++) {
        int bo_tgt = m_nodes.bo_target.at(n);
        if (bo_tgt > 0)
            fprintf(file, "T %d %d %s %s %g\n", n, bo_tgt, bo_symbol.c_str(),
                    bo_symbol.c_str(), m_nodes.bo_score.at(n));
        int limit = m_nodes.limit_arc.at(n);
        if (limit == 0)
            continue;
        int first = m_nodes.limit_arc.at(n-1);
        assert(first <= limit);
        for (int a = first; a < limit; a++) {
            int tgt = m_arcs.target.at(a);
            string symbol = m_symbol_map.at(m_arcs.symbol.at(a));
            float score = m_arcs.score.at(a);
            fprintf(file, "T %d %d %s %s %g\n", n, tgt,
                    symbol.c_str(), symbol.c_str(), score);
        }
    }
}

void LM::write_fsmt_node(FILE *file, int n, string bo_symbol) const
{
    int bo_tgt = m_nodes.bo_target.at(n);
    if (bo_tgt > 0)
        fprintf(file, "%d %d %s %g\n", n, bo_tgt,
                bo_symbol.c_str(), -log10_to_ln(m_nodes.bo_score.at(n)));
    int limit = m_nodes.limit_arc.at(n);
    if (limit == 0)
        return;
    int first = m_nodes.limit_arc.at(n-1);
    assert(first <= limit);
    for (int a = first; a < limit; a++) {
        int tgt = m_arcs.target.at(a);
        if (n == m_empty_node_id && tgt == m_initial_node_id) {
            fprintf(stderr, "WARNING: omitting sentence start arc\n");
            continue;
        }
        string symbol = m_symbol_map.at(m_arcs.symbol.at(a));
        float score = -log10_to_ln(m_arcs.score.at(a));
        fprintf(file, "%d %d %s %g\n", n, tgt, symbol.c_str(), score);
    }
}

void LM::write_fsmt(FILE *file, string bo_symbol) const
{
    write_fsmt_node(file, m_initial_node_id, bo_symbol);
    for (int n = 1; n < num_nodes(); n++) {
        if (n == m_initial_node_id)
            continue;
        write_fsmt_node(file, n, bo_symbol);
    }
    fprintf(file, "%d\n", m_final_node_id);
}

void LM::fetch_probs(int node_id, vector<float> &vec)
{
    vec.clear();
    vec.resize(m_symbol_map.size(), FLT_MAX);
    assert(node_id != m_final_node_id);
    float bo_score = 0;
    while (1) {
        int limit = m_nodes.limit_arc.at(node_id);
        assert(limit > 0);
        int first = m_nodes.limit_arc.at(node_id - 1);
        assert(first < limit);
        for (int a = first; a < limit; a++) {
            int symbol = m_arcs.symbol.at(a);
            if (m_non_event[symbol])
                continue;
            if (vec[symbol] < FLT_MAX)
                continue;
            float score = m_arcs.score.at(a) + bo_score;
            vec[symbol] = score;
        }
        if (node_id == m_empty_node_id)
            break;
        bo_score += m_nodes.bo_score.at(node_id);
        node_id = m_nodes.bo_target.at(node_id);
    }
}

string LM::debug_str() const
{
    string str;
    for (int n = 0; n < num_nodes(); n++) {
        str.append(str::fmt(256, "%d bs=%g bt=%d l=%d\n",
                            n, m_nodes.bo_score.at(n),
                            m_nodes.bo_target.at(n), m_nodes.limit_arc.at(n)));
    }
    for (int a = 0; a < num_arcs(); a++) {
        str.append(str::fmt(256, "%d s=%d t=%d s=%g\n", a, m_arcs.symbol.at(a),
                            m_arcs.target.at(a), m_arcs.score.at(a)));
    }
    return str;
}

bool LM::debug_check_sum(int node_id) const
{
    double sum = 0;
    for (int i = 0; i < m_symbol_map.size(); i++) {
        if (m_non_event.at(i))
            continue;
        float score = 0;
        walk(node_id, i, &score);
        sum += pow(10, score);
    }
    bool ok = (fabs(sum - 1.0) < 1e-3);
    if (!ok)
        fprintf(stderr, "SUM %d = %g\n", node_id, sum);
    return  ok;
}

bool LM::debug_check_sums() const
{
    bool ok = true;
    for (int n = 1; n < num_nodes(); n++)
        ok = ok & debug_check_sum(n);
    return ok;
}

bool LM::debug_check_zero_bo() const
{
    bool ok = true;
    for (int n = 0; n < num_nodes(); n++) {
        if (num_children(n) > 0)
            continue;
        float bo_score = m_nodes.bo_score.at(n);
        if (bo_score != 0) {
            fprintf(stderr, "WARNING: node %d has no children but bo_score = %g\n",
                    n, bo_score);
            ok = false;
        }
    }
    return ok;
}

};
