#include <map>
#include <stdlib.h>
#include "str.hh"
#include "Lattice.hh"

Lattice::Arc::Arc(int target_node_id, std::string label, float ac_log_prob,
		  float lm_log_prob)
  : target_node_id(target_node_id), 
    label(label), 
    ac_log_prob(ac_log_prob), 
    lm_log_prob(lm_log_prob)
{ 
}

Lattice::Lattice()
  : initial_node_id(-1), final_node_id(-1), m_num_arcs(0)
{
}

void
Lattice::clear()
{
  m_nodes.clear();
  m_num_arcs = 0;
  initial_node_id = -1;
  final_node_id = -1;
}

Lattice::Node&
Lattice::new_node()
{
  m_nodes.push_back(Node(m_nodes.size()));
  return m_nodes.back();
}

void
Lattice::new_arc(int S, int E, std::string W, float a, float l)
{
  m_nodes.at(S).arcs.push_back(Arc(E, W, a, l));
  m_num_arcs++;
}



void
Lattice::read(FILE *file)
{
  m_nodes.clear();
  m_num_arcs = 0;
  final_node_id = -1;
  initial_node_id = -1;

  std::map<int, int> label_map;
  std::string line;
  std::vector<std::string> fields;
  std::vector<std::string> attribute;
  while (str::read_line(&line, file, true)) {

    // Remove comments and empty lines
    str::clean_after(&line, "#");
    str::clean(&line, " \t");
    if (line.empty())
      continue;

    // Parse fields
    str::split(&line, " \t", true, &fields);
    bool ok = true;

    {
      // Node (only node label supported)
      if (fields[0][0] == 'I') {
	str::split(&fields[0], "=", false, &attribute, 2);
	int node_label = str::str2long(&attribute[1], &ok);
	Node &node = new_node();
	label_map[node_label] = node.id;
      }

      // Arc (supports only S, E, W, a, l)
      else if (fields[0][0] == 'J') {
	int S = -1;
	int E = -1;
	std::string W;
	float a = 0;
	float l = 0;

	while (!fields.empty()) {
	  str::split(&fields.back(), "=", false, &attribute, 2);
	  fields.pop_back();
	  if (attribute[0] == "S")
	    S = str::str2long(&attribute[1], &ok);
	  else if (attribute[0] == "E")
	    E = str::str2long(&attribute[1], &ok);
	  else if (attribute[0] == "W")
	    W = attribute[1];
	  else if (attribute[0] == "a")
	    a = str::str2float(&attribute[1], &ok);
	  else if (attribute[0] == "l")
	    l = str::str2float(&attribute[1], &ok);
	}
	S = label_map[S];
	E = label_map[E];
	new_arc(S, E, W, a, l);
      }

      else {
	while (!fields.empty()) {
	  str::split(&fields.back(), "=", false, &attribute, 2);
	  fields.pop_back();
	  if (attribute[0] == "start")
	    initial_node_id = str::str2long(&attribute[1], &ok);
	  else if (attribute[0] == "end")
	    final_node_id = str::str2long(&attribute[1], &ok);
	}
      }

      if (!ok) {
	fprintf(stderr, "ERROR: invalid attributes in SLF file:\n%s\n",
		line.c_str());
	exit(1);
      }
    }
  }

  if (initial_node_id < 0 || final_node_id < 0) {
    fprintf(stderr, "ERROR: start and end not specified in SLF file\n");
    exit(1);
  }

  initial_node_id = label_map[initial_node_id];
  final_node_id = label_map[final_node_id];
}

void
Lattice::write(FILE *file)
{
  fprintf(file, "VERSION=1.1\n"
	  "base=10\n"
	  "start=%d end=%d\n"
	  "N=%d L=%d\n", initial_node_id, final_node_id, 
	  m_nodes.size(), m_num_arcs);
  
  for (int n = 0; n < (int)m_nodes.size(); n++)
    fprintf(file, "I=%d\n", m_nodes[n].id);

  int J = 0;
  for (int n = 0; n < (int)m_nodes.size(); n++) {
    for (int a = 0; a < (int)m_nodes[n].arcs.size(); a++) {
      Arc &arc = m_nodes[n].arcs[a];
      fprintf(file, "J=%d S=%d E=%d W=%s a=%e l=%e\n",
	      J, n, arc.target_node_id, arc.label.c_str(), arc.ac_log_prob,
	      arc.lm_log_prob);
      J++;
    }
  }
}
