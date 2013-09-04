#ifndef FST_HH
#define FST_HH

#include <vector>
#include <string>
#include <sstream>

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

    inline std::string str() {
      std::ostringstream os;
      os << "Arc " << source << " -> " << target << " (" << transition_logprob << "): " << emit_symbol;
      return os.str();
    }
  };
    
  struct Node {
    Node() : emission_pdf_idx(-1), end_node(false), pruned(false) {}
    int emission_pdf_idx;
    std::vector<int> arcidxs;
    bool end_node;
    bool pruned;

    inline std::string str() {
      std::ostringstream os;
      os << "Node " << emission_pdf_idx << " (" << arcidxs.size() << ")";
      return os.str();
    }
  };

  Fst();
  void read(std::string &);
  inline void read(const char *s) {std::string ss(s); read(ss);}
  int initial_node_idx;
  std::vector<Node> nodes;
  std::vector<Arc> arcs;
};

#endif

