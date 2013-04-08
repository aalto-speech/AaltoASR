#ifndef TRIE_HH
#define TRIE_HH

#include <cstddef>  // NULL
#include <algorithm>
#include <vector>

namespace misc {

  template<typename Sym, typename Value, Value default_value = 0>
  class Trie {
  public:

    class Arc {
    public:
      Arc() : sym(0), tgt(0) { }
      Arc(Sym sym) : sym(sym), tgt(0) { }
      Arc(Sym sym, int tgt) : sym(sym), tgt(tgt) { }

      Sym sym;
      int tgt;

      bool operator<(const Arc &arc) const 
      {
        return sym < arc.sym;
      }
    };

    typedef std::vector<Arc> ArcVec;

    class Node {
    public:
      ArcVec arcs;
      Value value;

      Node() : value(default_value) { }

      /** Tries to insert a new arc with given symbol.  
       *
       * \param sym = symbol to insert
       *
       * \param inserted = if non-null, set to true if insertion
       * succeeds, and false if symbol exists already
       */
      typename ArcVec::iterator insert(Sym sym, bool *inserted)
      {
        Arc arc(sym, 0);

        typename ArcVec::iterator it = 
          std::lower_bound(arcs.begin(), arcs.end(), arc);
        if (inserted != NULL)
          *inserted = true;

        if (it == arcs.end() || it->sym != arc.sym)
          return arcs.insert(it, arc);

        if (inserted != NULL)
          *inserted = false;
        return it;
      }

      /** Find arc with the given symbol.
       * \param sym = symbol to find
       * \return iterator to arc vector
       */
      typename ArcVec::iterator find(Sym sym)
      {
        typename ArcVec::iterator it = 
          std::lower_bound(arcs.begin(), arcs.end(), Arc(sym));
        if (it == arcs.end() || it->sym != sym)
          return arcs.end();
        return it;
      }

    };

    Trie() 
    {
      clear();
    }

    /** Clear the structure. */
    void clear()
    {
      m_nodes.clear();
      m_nodes.resize(1);
      m_root = 0;
    }

    /** Insert a symbol to a given node.  Does not insert if exists already.
     * \param n = node to insert the symbol to
     * \param sym = symbol to insert
     * \return the new or existing node
     */
    int insert(int n, Sym sym)
    {
      bool inserted = false;
      typename ArcVec::iterator it = m_nodes.at(n).insert(sym, &inserted);
      if (!inserted)
        return it->tgt;
      int tgt = it->tgt = m_nodes.size();
      m_nodes.push_back(Node());
      return tgt;
    }

    /** Find a target node for the given symbol.
     * \param n = node to search from
     * \param sym = symbol to search
     * \return target node index or negative if not found
     */
    int find(int n, Sym sym)
    {
      Node &node = m_nodes.at(n);
      typename ArcVec::iterator it = node.find(sym);
      if (it == node.arcs.end())
        return -1;
      return it->tgt;
    }

    /** Root node index. */
    int root() const 
    {
      return m_root;
    }

    /** Access a node. */
    Node &node(int n)
    {
      return m_nodes.at(n);
    }

  private:
    
    std::vector<Node> m_nodes;
    int m_root;

  };

};

#endif /* TRIE_HH */
