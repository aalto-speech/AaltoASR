#ifndef FST_HH
#define FST_HH

#include <vector>
#include <string>

class Fst {
public:
  struct ReadError : public std::exception {
    virtual const char *what() const throw() {
      return "Fst: read error"; }
  };

  struct Arc {
    int source;
    int target;
    float transition_logprob;
    std::string emit_symbol;
  };
    
  struct Node {
    Node() : emission_pdf_idx(-1), end_node(false), pruned(false) {}
    int emission_pdf_idx;
    std::vector<Arc *> arcptrs;
    bool end_node;
    bool pruned;
  };

  Fst();
  void read(std::string &);
  inline void read(const char *s) {std::string ss(s); read(ss);}
  int initial_node_idx;
  std::vector<Node> nodes;
  std::vector<Arc> arcs;
};

#endif

