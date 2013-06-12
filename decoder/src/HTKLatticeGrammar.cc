#include "HTKLatticeGrammar.hh"
#include "misc/str.hh"

void HTKLatticeGrammar::read_error(std::string &s) {
  fprintf(stderr,"HTKDialogLattice read(), error in line %d: '%s'\n",
	 m_lineno, s.c_str());
  exit(-1);
}

bool HTKLatticeGrammar::read_clean_line(std::string &str, FILE *file, bool b)
{
  bool ok = str::read_line(str, file, b);

  // Remove windows-formatted line feeds
  if (str[str.length() - 1] == '\r')
    str.resize(str.length() - 1);
  return ok;
}

void HTKLatticeGrammar::read(FILE *file, bool binary) {
  assert(binary==false);
  std::string line;
  std::vector<std::string> vec, vec2;

  // Just for efficiency
  line.reserve(128); 
  vec.reserve(16);
  vec2.reserve(16);

  bool ok = true;

  m_lineno = 0;

  // First, read fixed headers.
  ok = read_clean_line(line, file, true);
  m_lineno++;

  if (!ok || line != "VERSION=1.0") {
    read_error(line);
  }

  ok = read_clean_line(line, file, true);
  m_lineno++;
  if (!ok || line != "lmscale=1.00") {
    read_error(line);
  }

  ok = read_clean_line(line, file, true);
  m_lineno++;
  if (!ok || line != "wdpenalty=0.00")  {
    read_error(line);
  }

  // Read the number of nodes and the number of arcs
  ok = read_clean_line(line, file, true);
  m_lineno++;
  str::clean(line, " \t");
  vec = str::split(line, " \t", true);
  ok = (ok && vec.size()==2);
  //fprintf(stderr,"'%s' '%s'\n", vec[0].c_str(), vec[1].c_str());
  
  int num_nodes, num_arcs;
  ok = ok && sscanf(vec[0].c_str(),"N=%d",&num_nodes);
  ok = ok && sscanf(vec[1].c_str(),"L=%d",&num_arcs);
  if (!ok) read_error(line);
  //fprintf(stderr,"%d %d\n", num_nodes, num_arcs);

  // Read nodes
  m_nodes.resize(num_nodes);
  for (int i=0; i<num_nodes; i++) {
    int x;
    ok = read_clean_line(line, file, true);
    m_lineno++;
    str::clean(line, " \t");
    vec = str::split(line, " \t", true);
    ok = ok && vec.size()==2;
    ok = ok && sscanf(vec[0].c_str(),"I=%d",&x);
    ok = ok && x == i; 
    ok = ok && vec[1]=="t=0.00";
    if (!ok) read_error(line);
  }

  // Read arcs
  m_arcs.resize(num_arcs);
  char readbuf[1000];
  for (int i=0; i<num_arcs; i++) {
    int x;
    Arc &cur_arc=m_arcs[i];
    ok = read_clean_line(line, file, true);
    m_lineno++;
    str::clean(line, " \t");
    vec = str::split(line, " \t", true);
    ok = ok && vec.size()==7;
    ok = ok && sscanf(vec[0].c_str(),"J=%d",&x);
    ok = ok && x == i;
    ok = ok && sscanf(vec[1].c_str(),"S=%d",&x);
    cur_arc.source=x;
    ok = ok && sscanf(vec[2].c_str(),"E=%d",&x);
    cur_arc.target=x;
    vec2 = str::split(vec[3], "=", false);
    ok = ok && vec2.size()==2;

    // Replace sentence ends and sentence starts with 
    // with the internal representation
    std::string word = vec2[1];
    if (word == "!SENT_START" ) {
      word = "<s>";
      cur_arc.widx = Vocabulary::add_word(word);
      m_start_node_idx = cur_arc.source;
    } else if (word == "!SENT_END" ) {
      word = "</s>";
      cur_arc.widx = Vocabulary::add_word(word);
      m_end_node_idx = cur_arc.target;
    } else {
      cur_arc.widx = Vocabulary::add_word(word);
      if (word == "!NULL" ) 
	m_null_idx = cur_arc.widx;
    }

    ok = ok && vec[4] == "v=0";
    ok = ok && vec[5] == "a=0.00";
    ok = ok && vec[6] == "l=0.000";

    if (!ok) read_error(line);

    // Mark the arc to the corresponding node
    m_nodes[cur_arc.source].arcs_out.push_back(i);
    /*fprintf(stdout,"J=%d\tS=%d\tE=%d\tW=%d\n",
      i, cur_arc.source, cur_arc.target, cur_arc.widx);*/
  }
  /*fprintf(stdout,"sidx %d, eidx %d, nidx %d\n", m_start_node_idx,
    m_end_node_idx, m_null_idx);*/
  
  // For lookahead
  pregenerate_bigram_idxlist();
}

void HTKLatticeGrammar::write(FILE *file, bool binary) {
  assert(binary==false);
  fprintf(file,"VERSION=1.0\nlmscale=1.00\nwdpenalty=0.00}n");
  fprintf(file,"N=%ld L=%ld\n",m_nodes.size(),m_arcs.size());
  for (int i=0;i<m_nodes.size();i++) {
    fprintf(file,"I=%d\tt=0.00\n",i);
  }
  for (int i=0;i<m_arcs.size();i++) {
    Arc cur_arc=m_arcs[i];
    fprintf(file,"J=%d\tS=%d\tE=%d\tw=%s\tv=0\ta=0.00\tl=0.000\n",
	    i, cur_arc.source, cur_arc.target, word(cur_arc.widx).c_str()
	    );
  }
}

bool HTKLatticeGrammar::match_begin(const Gram &g) {
  std::vector<std::vector<int> > active_states(2);
  active_states[0].push_back(m_start_node_idx);
  bool retval=true;
  
  // Start consuming input
  for (int cur_token_num = 0; cur_token_num<g.size(); cur_token_num++) {
    retval=true;
    std::vector<int> &cur_states = active_states[cur_token_num%2];
    std::vector<int> &new_states = active_states[(cur_token_num+1)%2];

#if 0
    fprintf(stdout,"Active nodes:\n");
    for (int i=0;i<cur_states.size();i++) {
      fprintf(stdout,"%d (",cur_states[i]);
      Node n = m_nodes[cur_states[i]];
      for (int j=0;j<n.arcs_out.size(); j++) {
	fprintf(stdout,"%d ", n.arcs_out[j]);
      }
      fprintf(stdout,")\n");
    }
#endif

    if (cur_states.size()==0) return false;
    new_states.clear();
    const int cur_widx = g[cur_token_num];
    for (int node_num=0; node_num<cur_states.size(); node_num++) {
      int node_idx = cur_states[node_num];
      Node n = m_nodes[node_idx];
      for (int arc_idx=0; arc_idx<n.arcs_out.size(); arc_idx++) {
	Arc a = m_arcs[n.arcs_out[arc_idx]];

	//fprintf(stdout,"nullidx=%d=%d\n", m_null_idx, word_index("!NULL"));
	/*fprintf(stdout,"widx=%d(%s), Looking at node %d, arc %d (%d:%s)\n",
		cur_widx, word(cur_widx).c_str(),node_idx,
		arc_idx, a.widx, word(a.widx).c_str());*/
	if (a.widx == m_null_idx) { // Take all NULL transitions
	  //fprintf(stdout,"Going null, pushing new astate %d\n",a.target);
	  cur_states.push_back(a.target);
	  continue;
	}

	if (a.widx == cur_widx) {
	  //fprintf(stderr,"Going new, pushing %d\n",a.target);
	  new_states.push_back(a.target);
	  continue;
	}
      }
    }
    if (new_states.size()==0) retval=false;
  }
  return retval;
}

bool HTKLatticeGrammar::match_begin(const std::string &string_in) {
  std::vector<std::string> vec;
  std::string s = string_in;
  str::clean(s, " \t");
  vec = str::split(s, " \t", true);

  Gram g;
  for (int i=0; i<vec.size();i++) {
    g.push_back(word_index(vec[i]));
  }

  return match_begin(g);
}

void HTKLatticeGrammar::pregenerate_bigram_idxlist() {
  m_bigram_idxlist.clear();
  m_bigram_idxlist.resize(m_words.size());

  int null_idx = word_index("!NULL");

  for (std::vector<Arc>::iterator abeg=m_arcs.begin(); abeg!=m_arcs.end(); ++abeg) {
    if (abeg->widx == null_idx) continue;
    //fprintf(stderr, "Gen for (%s %d):\n", word(abeg->widx).c_str(), abeg->widx);
    std::vector<int> &target_arclist = m_nodes[abeg->target].arcs_out;
    std::set<int> &bigram_targetset = m_bigram_idxlist.at(abeg->widx);
    for (std::vector<int>::iterator atarget=target_arclist.begin();
         atarget!=target_arclist.end(); ++atarget) {
      Arc *target = &m_arcs[*atarget];

      // Skip NULL nodes
      while (target->widx == null_idx) {
        Node nextnode = m_nodes[target->target];
        assert(nextnode.arcs_out.size()==1);
        target = &(m_arcs[nextnode.arcs_out[0]]);
      }
      //fprintf(stderr, " (%s %d)", word(target->widx).c_str(), target->widx);
      bigram_targetset.insert(target->widx);
    }
    //fprintf(stderr,"\n");
  }
}


void HTKLatticeGrammar::fetch_bigram_list(int prev_word_id, 
                                          std::vector<float> &result_buffer) {
  result_buffer.resize(m_words.size());
  for (std::vector<float>::iterator it=result_buffer.begin(); it!=result_buffer.end(); ++it) {
    *it = IMPOSSIBLE_LOGPROB;
  }

  std::set<int> &possible_targets = m_bigram_idxlist[prev_word_id];
  //fprintf(stderr, "%ld possible LA continuations for %s:", m_bigram_idxlist[prev_word_id].size(), 
  //        word(prev_word_id).c_str());
  for (std::set<int>::iterator idx = possible_targets.begin();
       idx != possible_targets.end(); ++idx) {
    //fprintf(stderr, " %d", *idx);
    result_buffer[*idx] = 0.0f;
  }
  //fprintf(stderr, "\n");
}
