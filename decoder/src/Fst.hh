#ifndef FST_HH
#define FST_HH
/* 
   Simple class to handle mitfst (http://people.csail.mit.edu/ilh/fst/) format networks. 
   AT&T fst toolkit and openfst have very similar formats, so this may work directly or
   with small adjustments with thosenetworks.
*/

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
    Node() : emission_pdf_idx(-1), end_node(false) {}
    int emission_pdf_idx;
    std::vector<int> arcidxs;
    bool end_node;

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

