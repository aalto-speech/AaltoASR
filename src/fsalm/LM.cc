#include <math.h>
#include "fsalm/algo.hh"
#include "fsalm/LM.hh"
#include "fsalm/ArpaReader.hh"
#include "misc/macros.hh"
#include <deque>

namespace fsalm {

  static MaxPlusSemiring maxplus_semiring;

  float log10_to_ln(float f)
  {
    return f * M_LN10;
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
    int limit = m_nodes.limit_arc.get(node_id);
    if (limit == 0)
      return 0;
    int first = m_nodes.limit_arc.get(node_id - 1);
    assert(limit >= first);
    return limit - first;
  }

  Str LM::str(const IntVec &vec) const
  {
    Str ret;
    FOR(i, vec) {
      if (i > 0)
        ret.append(" ");
      ret.append(m_symbol_map.at(vec[i]));
    }
    return ret;
  }

  int LM::find_backoff(IntVec vec) const
  {
    if (vec.empty())
      return m_empty_node_id;
    if (vec.back() == m_end_symbol)
      throw Error("LM::find_backoff(): sentence end not allowed");

    while (1) {
      IntVec nodes = walk_no_bo_vec(m_empty_node_id, vec);
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
      throw Error("LM::find_child(): final node not allowed");
    int limit = m_nodes.limit_arc.get(node_id);
    if (limit > 0) {
      int first = m_nodes.limit_arc.get(node_id - 1);
      assert(limit >= first); 
      if (limit > first) {
        int arc_id = fsalm::binary_search<Array,int>(m_arcs.symbol, symbol, first, limit);
        if (arc_id != limit) {
          if (score != NULL)
            *score += m_arcs.score.get(arc_id);
          return m_arcs.target.get(arc_id);
        }
      }
    }
    return -1;
  }
  
  int LM::walk_no_bo(int node_id, const IntVec &vec, float *score) const
  {
    FOR(i, vec) {
      node_id = walk_no_bo(node_id, vec[i], score);
      if (node_id < 0)
        return -1;
    }
    return node_id;
  }

  IntVec LM::walk_no_bo_vec(int node_id, const IntVec &vec, float *score) const
  {
    IntVec ret;
    ret.reserve(vec.size());
    FOR(i, vec) {
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
          *score += m_nodes.bo_score.get(node_id);
        node_id = m_nodes.bo_target.get(node_id);
        continue;
      }
      return new_node_id;
    }        
  }

  int LM::walk(int node_id, const IntVec &vec, float *score) const
  {
    FOR(i, vec) {
      node_id = walk(node_id, vec[i], score);
    }
    return node_id;
  }

  void LM::new_arc(int src_node_id, int symbol, int tgt_node_id, float score)
  {
    assert(src_node_id > 0);
    int arc_id = num_arcs();
    int limit_arc = m_nodes.limit_arc.get(src_node_id);
    assert(limit_arc == 0 || limit_arc == arc_id);

    // Set limit_arc for nodes that precede src_node_id and have
    // limit_arc unset.
    if (limit_arc == 0 && arc_id > 0) {
      for (int n = src_node_id - 1; n > 0; n--) {
        if (m_nodes.limit_arc.get(n) > 0)
          break;
        m_nodes.limit_arc.set_grow_widen(n, arc_id);
      }
    }

    m_nodes.limit_arc.set_grow_widen(src_node_id, arc_id + 1);
    set_arc(arc_id, symbol, tgt_node_id, score);
  }

  void LM::new_ngram(const IntVec &vec, float score, float bo_score)
  {
    assert(!vec.empty());

    // Find the context node index, by checking if it is the same as
    // on the previous insertion.
    //
    IntVec ctx_vec(vec.begin(), vec.end() - 1);
    if (m_cache.ctx_node_id < 0 || ctx_vec != m_cache.ctx_vec) {      
      m_cache.ctx_node_id = walk_no_bo(empty_node_id(), ctx_vec);
      if (m_cache.ctx_node_id < 0)
        throw Error("prefix missing for ngram \"" + str(vec) + "\"");
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
      bo_node_id = find_backoff(IntVec(vec.begin() + 1, vec.end()));
      assert(bo_node_id >= 0);

      if (vec.size() == order()) {
        tgt_node_id = bo_node_id;
        assert(bo_score == 0);
      }
      else
        tgt_node_id = new_node();
    }
    if (vec.size() == 1 && vec.back() == m_start_symbol)
      m_initial_node_id = tgt_node_id;

    // Create arc and update possible backoff information.
    //
    new_arc(m_cache.ctx_node_id, vec.back(), tgt_node_id, score);
    if (tgt_node_id != bo_node_id) {
      m_nodes.bo_target.set_grow_widen(tgt_node_id, bo_node_id);
      m_nodes.bo_score.set_grow_widen(tgt_node_id, bo_score);
    }
  }

  int LM::new_node()
  {
    int node_id = num_nodes();
    m_nodes.bo_target.set_grow_widen(node_id, 0);
    m_nodes.bo_score.set_grow_widen(node_id, 0);
    m_nodes.limit_arc.set_grow_widen(node_id, 0);
    return node_id;
  }

  void LM::set_arc(int arc_id, int symbol, int target, float score)
  {
    m_arcs.symbol.set_grow_widen(arc_id, symbol);
    m_arcs.target.set_grow_widen(arc_id, target);
    m_arcs.score.set_grow_widen(arc_id, score);
  }

  void LM::trim()
  {
    // Find childless nodes and compute new node indices by not
    // counting childless nodes and backoffing for removed nodes.
    //
    IntVec new_target(num_nodes(), 0);
    BoolVec removed(num_nodes(), false);
    new_target.at(0) = 0;
    int new_n = 1;
    for (int n = 1; n < num_nodes(); n++) {
      if (num_children(n) == 0) {
        float bo_score = m_nodes.bo_score.get(n);
        if (bo_score != 0)
          fprintf(stderr, "WARNING: LM::trim(): childless node %d "
                  "with bo_score = %g\n", n, bo_score);

//          throw Error(str::fmt(256, "LM::trim(): childless node %d "
//                               "with bo_score = %g", n, bo_score));
        new_target[n] = new_target.at(m_nodes.bo_target.get(n));
        removed[n] = true;
      }
      else
        new_target[n] = new_n++;
    }

    // Correct arc target indices.  Replace childless targets by
    // backoff targets.
    //
    for (int a = 0; a < num_arcs(); a++) {
      int tgt = m_arcs.target.get(a);
      m_arcs.target.set(a, new_target.at(tgt));
    }

    // Remove childless nodes and update backoff target indices.
    //
    for (int n = 1; n < num_nodes(); n++) {
      if (removed[n])
        continue;
      m_nodes.bo_score.set(new_target[n], m_nodes.bo_score.get(n));
      m_nodes.bo_target.set(new_target[n], 
                            new_target.at(m_nodes.bo_target.get(n)));
      m_nodes.limit_arc.set(new_target[n], m_nodes.limit_arc.get(n));
    }
    m_nodes.bo_score.resize(new_n);
    m_nodes.bo_target.resize(new_n);
    m_nodes.limit_arc.resize(new_n);

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

  void LM::compute_potential(FloatVec &d)
  {
    // Implemented according to "Mehryar Mohri and Michael Riley.  A
    // Weight Pushing Algorithm for Large Vocabulary Speech
    // Recognition.  Eurospeech 2001."

    // Compute incoming arcs for each node.
    std::vector<std::vector<InArc> > in_arcs;
    in_arcs.resize(num_nodes());
    for (int n = 0; n < num_nodes(); n++) {
      int bo_tgt = m_nodes.bo_target.get(n);
      if (bo_tgt > 0)
        in_arcs.at(bo_tgt).push_back(InArc(n, -1));

      int limit = m_nodes.limit_arc.get(n);
      if (limit == 0)
        continue;
      int first = m_nodes.limit_arc.get(n-1);
      assert(first <= limit);
      for (int a = first; a < limit; a++) {
        int tgt = m_arcs.target.get(a);
        in_arcs.at(tgt).push_back(InArc(n, a));
      }
    }

    int N = num_nodes();
    d.clear();
    d.resize(N, semiring->zero);
    FloatVec r(N, semiring->zero);
    BoolVec in_queue(N, false);
    std::deque<int> queue;

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

      std::vector<InArc> &in_arcs_q = in_arcs.at(q);
      FOR(a, in_arcs_q) {
        InArc &in_arc = in_arcs_q[a];
        int n = in_arc.source;

        float score = 0;
        if (in_arc.arc_id < 0)
          score = m_nodes.bo_score.get(n);
        else
          score = m_arcs.score.get(in_arc.arc_id);

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
    FloatVec potential;
    compute_potential(potential);

    assert(potential.size() == num_nodes());
    assert(potential.at(m_final_node_id) == semiring->one);
    
    m_final_score = potential.at(m_initial_node_id);

    for (int n = 0; n < num_nodes(); n++) {
      int bo_tgt = m_nodes.bo_target.get(n);
      if (bo_tgt > 0) {
        float score = m_nodes.bo_score.get(n);
        score = semiring->times(score, potential.at(bo_tgt));
        score = semiring->divide(score, potential.at(n));
        m_nodes.bo_score.set(n, score);
      }

      int limit = m_nodes.limit_arc.get(n);
      if (limit == 0)
        continue;
      int first = m_nodes.limit_arc.get(n-1);
      assert(first <= limit);
      for (int a = first; a < limit; a++) {
        float score = m_arcs.score.get(a);
        int target = m_arcs.target.get(a);
        score = semiring->times(score, potential.at(target));
        score = semiring->divide(score, potential.at(n));
        m_arcs.score.set(a, score);
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
    catch (Ex &e) {
      throw Error(Str("LM::read_arpa(): error while reading unigrams: ") 
                  + e.what());
    }

    try {
      m_start_symbol = m_symbol_map.index(start_str);
      m_end_symbol = m_symbol_map.index(end_str);
    }
    catch (std::string &str) {
      throw Error(
        "LM::read_arpa(): sentence start '" + start_str + 
        "' or sentence end '" + end_str + "' not in unigrams");
    }      

    for (size_t i = 0; i < reader.order_ngrams.size(); i++) {
      ArpaReader::Ngram &ngram = reader.order_ngrams[i];
      assert(ngram.symbols.size() == 1);
      new_ngram(ngram.symbols, ngram.log_prob, ngram.backoff);
    }

    // Continue from 2-grams 
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
      throw Error("LM::read() error while reading header");
    str::read_line(start_str, file, true);
    str::read_line(end_str, file, true);
    m_symbol_map.read(file);
    m_arcs.symbol.read(file);
    m_arcs.target.read(file);
    m_arcs.score.read(file);
    m_nodes.bo_score.read(file);
    m_nodes.bo_target.read(file);
    m_nodes.limit_arc.read(file);

    m_start_symbol = m_symbol_map.index(start_str);
    m_end_symbol = m_symbol_map.index(end_str);

    m_non_event.clear();
    m_non_event.resize(m_symbol_map.size(), false);
    m_non_event.at(m_start_symbol) = true;

    // Check that initial node matches start symbol
    int node_id = walk(m_empty_node_id, m_start_symbol);
    if (node_id != m_initial_node_id)
      throw Error(str::fmt(1024, "LM::read(): initial node %d does not match "
                           "start symbol target %d", m_initial_node_id,
                           node_id));
  }

  void LM::write(FILE *file) const
  {
    fprintf(file, "LM1:%d:%d:%d:%d:%g:",
            m_order, m_empty_node_id, m_initial_node_id, m_final_node_id, m_final_score);
    fprintf(file, "%s\n%s\n", start_str.c_str(), end_str.c_str());
    m_symbol_map.write(file);
    m_arcs.symbol.write(file);
    m_arcs.target.write(file);
    m_arcs.score.write(file);
    m_nodes.bo_score.write(file);
    m_nodes.bo_target.write(file);
    m_nodes.limit_arc.write(file);
  }

  void LM::write_fst(FILE *file, Str bo_symbol) const
  {
    fputs("#FSTBasic MaxPlus\n", file);
    fprintf(file, "I %d\n", m_initial_node_id);
    fprintf(file, "F %d\n", m_final_node_id);
    
    for (int n = 1; n < num_nodes(); n++) {
      int bo_tgt = m_nodes.bo_target.get(n);
      if (bo_tgt > 0)
        fprintf(file, "T %d %d %s %s %g\n", n, bo_tgt, bo_symbol.c_str(),
                bo_symbol.c_str(), m_nodes.bo_score.get(n));
      int limit = m_nodes.limit_arc.get(n);
      if (limit == 0)
        continue;
      int first = m_nodes.limit_arc.get(n-1);
      assert(first <= limit);
      for (int a = first; a < limit; a++) {
        int tgt = m_arcs.target.get(a);
        Str symbol = m_symbol_map.at(m_arcs.symbol.get(a));
        float score = m_arcs.score.get(a);
        fprintf(file, "T %d %d %s %s %g\n", n, tgt, 
                symbol.c_str(), symbol.c_str(), score);
      }
    }
  }

  void LM::write_fsmt_node(FILE *file, int n, Str bo_symbol) const
  {
    int bo_tgt = m_nodes.bo_target.get(n);
    if (bo_tgt > 0)
      fprintf(file, "%d %d %s %g\n", n, bo_tgt, 
              bo_symbol.c_str(), -log10_to_ln(m_nodes.bo_score.get(n)));
    int limit = m_nodes.limit_arc.get(n);
    if (limit == 0)
      return;
    int first = m_nodes.limit_arc.get(n-1);
    assert(first <= limit);
    for (int a = first; a < limit; a++) {
      int tgt = m_arcs.target.get(a);
      if (n == m_empty_node_id && tgt == m_initial_node_id) {
        fprintf(stderr, "WARNING: omitting sentence start arc\n");
        continue;
      }
      Str symbol = m_symbol_map.at(m_arcs.symbol.get(a));
      float score = -log10_to_ln(m_arcs.score.get(a));
      fprintf(file, "%d %d %s %g\n", n, tgt, symbol.c_str(), score);
    }
  }

  void LM::write_fsmt(FILE *file, Str bo_symbol) const
  {
    write_fsmt_node(file, m_initial_node_id, bo_symbol);
    for (int n = 1; n < num_nodes(); n++) {
      if (n == m_initial_node_id)
        continue;
      write_fsmt_node(file, n, bo_symbol);
    }
    fprintf(file, "%d\n", m_final_node_id);
  }

  void LM::fetch_probs(int node_id, FloatVec &vec)
  {
    vec.clear();
    vec.resize(m_symbol_map.size(), FLT_MAX);
    assert(node_id != m_final_node_id);
    float bo_score = 0;
    while (1) {
      int limit = m_nodes.limit_arc.get(node_id);
      assert(limit > 0);
      int first = m_nodes.limit_arc.get(node_id - 1);
      assert(first < limit);
      for (int a = first; a < limit; a++) {
        int symbol = m_arcs.symbol.get(a);
        if (m_non_event[symbol])
          continue;
        if (vec[symbol] < FLT_MAX)
          continue;
        float score = m_arcs.score.get(a) + bo_score;
        vec[symbol] = score;
      }
      if (node_id == m_empty_node_id)
        break;
      bo_score += m_nodes.bo_score.get(node_id);
      node_id = m_nodes.bo_target.get(node_id);
    }
  }

  Str LM::debug_str() const 
  {
    Str str;
    for (int n = 0; n < num_nodes(); n++) {
      str.append(str::fmt(256, "%d bs=%g bt=%d l=%d\n", 
                          n, m_nodes.bo_score.get(n), 
                          m_nodes.bo_target.get(n), m_nodes.limit_arc.get(n)));
    }
    for (int a = 0; a < num_arcs(); a++) {
      str.append(str::fmt(256, "%d s=%d t=%d s=%g\n", a, m_arcs.symbol.get(a),
                          m_arcs.target.get(a), m_arcs.score.get(a)));
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
      float bo_score = m_nodes.bo_score.get(n);
      if (bo_score != 0) {
        fprintf(stderr, "WARNING: node %d has no children but bo_score = %g\n",
                n, bo_score);
        ok = false;
      }
    }
    return ok;
  }

};
