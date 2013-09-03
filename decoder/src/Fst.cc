#include "Fst.hh"
#include "str.hh"

Fst::Fst(): initial_node_idx(-1) {
}

void Fst::read(std::string &fname) {
  std::string line;

  FILE *ifh = fopen(fname.c_str(), "r");
  if (ifh==NULL) {
    perror("Error");
    exit(-1); // FIXME: we should use exceptions
  }

  str::read_line(&line, ifh, true);
  if (line != "#FSTBasic MaxPlus") {
    fprintf(stderr, "Unknown header '%s'.\n", line.c_str());
    throw ReadError();
  }
  std::vector<std::string> fields;
  while (str::read_line(&line, ifh, true)) {
    str::split(&line, " ", true, &fields);
    if (fields.size()<2) {
      fprintf(stderr, "Too few fields '%s'.\n", line.c_str());
      throw ReadError();
    }

    // Resize nodes to the size of the first mentioned node
    auto first_node_idx = atoi(fields[1].c_str());
    if (nodes.size() <= first_node_idx) {
      nodes.resize(first_node_idx+1);
    }

    if (fields[0]=="I") {
      initial_node_idx = first_node_idx;
      if (fields.size()>2) {
        fprintf(stderr, "Too many fields for I: '%s'.\n", line.c_str());
        throw ReadError();
      }
      continue;
    }

    if (fields[0]=="F") {
      nodes[first_node_idx].end_node = true;
      if (fields.size()>2) {
        fprintf(stderr, "Too many fields for F: '%s'.\n", line.c_str());
        throw ReadError();
      }
      continue;
    }
    
    if (fields[0]=="T") {
      if (fields.size()<3 or fields.size()>6) {
        fprintf(stderr, "Weird number of fields for T: '%s'.\n", line.c_str());
        throw ReadError();
      }
      
      auto second_node_idx = atoi(fields[2].c_str());
      if (nodes.size() <= second_node_idx) {
        nodes.resize(second_node_idx+1);
      }

      Arc a;
      a.source = first_node_idx;
      a.target = second_node_idx;
      if (fields.size()>=4) {
        a.emit_symbol = fields[4];
      }

      if (fields.size()>=6) {
        a.transition_logprob = strtof(fields[5].c_str(), NULL);
      }
      arcs.push_back(a);
      nodes[first_node_idx].arcptrs.push_back(&(arcs[arcs.size()-1]));

      // Move emission pdf indices from arcs to nodes
      auto emission_pdf_idx = atoi(fields[3].c_str());
      if (nodes[second_node_idx].emission_pdf_idx==-1) {
        nodes[second_node_idx].emission_pdf_idx = emission_pdf_idx;
      } else if (nodes[second_node_idx].emission_pdf_idx != emission_pdf_idx) {
        fprintf(stderr, "Conflicting emission_pdf_indices for node %d: %d != %d.\n",
                second_node_idx, nodes[second_node_idx].emission_pdf_idx, emission_pdf_idx);
        throw ReadError();
      }


    } else {
      fprintf(stderr, "Weird type indicator: '%s'.\n", fields[0].c_str());
      throw ReadError();
    }
    
  }
  fclose(ifh);

}
