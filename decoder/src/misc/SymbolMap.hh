#ifndef SYMBOLMAP_HH
#define SYMBOLMAP_HH

#include <cstddef>  // NULL
#include <sstream>
#include <map>
#include <string>
#include <vector>
#include "str.hh"

namespace misc {

/** An enumerated map of symbols. */
  template <typename S, typename I>
  class SymbolMap {
  public:
    SymbolMap()
    {
    }

    /** Clear the structure. */
    void clear()
    {
      m_symbols.clear();
      m_indices.clear();
    }

    /** Write symbol map in a file assuming that S = std::string. */
    void write(FILE *file) const
    {
      fprintf(file, "SYM:%zd:", m_symbols.size());
      for (size_t i = 0; i < m_symbols.size(); i++)
        fprintf(file, "%s\n", m_symbols[i].c_str());
    }

    /** Read symbol map from a file assuming that S = std::string. 
     * \throw std::string if fails
     */
    void read(FILE *file) 
    {
      m_symbols.clear();
      m_indices.clear();

      size_t size;
      int ret = fscanf(file, "SYM:%zd:", &size);
      if (ret != 1)
        throw std::string("bit::SymbolMap::read() failed reading header");
      std::string line;
      for (size_t i = 0; i < size; i++) {
        bool ok = str::read_line(line, file, true);
        if (!ok)
          throw std::string("bit::SymbolMap::read() failed reading symbol");
        insert_new(line);
      }
    }

    /** Insert a dummy index that does not have any mapping from symbol
     * to index. */
    I insert_dummy(const S &symbol = S())
    {
      m_symbols.push_back(symbol);
      return m_symbols.size() - 1;
    }
  
    /** Insert a symbol to the map but only if it does not exist yet.
     * Use insert_new() if reinsert should raise error.
     * 
     * \param symbol = symbol to insert
     * \param was_new = if non-null, set to true if new word and false if old
     * \return the index of the inserted or existing symbol
     */
    I insert(const S &symbol, bool *was_new = NULL)
    {
      std::pair<typename Map::iterator, bool> ret =
        m_indices.insert(typename Map::value_type(symbol, m_symbols.size()));
      if (ret.second)
        m_symbols.push_back(symbol);
      if (was_new != NULL)
        *was_new = ret.second;
      return ret.first->second;
    }

    /** Insert a symbol to the map raising an error if the symbol exists
     * already.
     *
     * \param symbol = symbol to insert
     * \return the index of the inserted symbol
     */
    I insert_new(const S &symbol)
    {
      std::pair<typename std::map<S, I>::iterator, bool> ret =
        m_indices.insert(
          typename std::map<S, I>::value_type(symbol, m_symbols.size()));
      if (!ret.second)
        throw std::string("tried to reinsert symbol");
      m_symbols.push_back(symbol);
      return ret.first->second;
    }
  
    /** Return the index of the symbol raising an error if the symbol
     * does not exist. */
    I index(const S &symbol) const
    {
      typename Map::const_iterator it = m_indices.find(symbol);
      if (it == m_indices.end())
        throw std::string("symbol not in the symbol map");
      return it->second;
    }

    /** Return the index of the symbol returning negative if symbol
     * does not exist. */
    I index_nothrow(const S &symbol) const
    {
      typename Map::const_iterator it = m_indices.find(symbol);
      if (it == m_indices.end())
        return -1;
      return it->second;
    }

    /** Return the symbol with the given index. */
    const S &at(I index) const
    {
      return m_symbols.at(index);
    }

    /** Return the symbol with the given index. */
    S &at(I index) 
    {
      return m_symbols.at(index);
    }

    /** Return the size of the map. */
    I size() const { return m_symbols.size(); }

    /** Return a vector of symbols as a string. */
    std::string str(const std::vector<I> &vec) const 
    {
      std::ostringstream out;
      for (size_t i = 0; i < vec.size(); i++) {
        if (i > 0)
          out << " ";
        out << vec[i];
      }
      return out.str();
    }

  private:
    typedef std::map<S, I> Map;
    std::vector<S> m_symbols; //!< Symbols in the map
    Map m_indices; //!< Indices of the symbols
  };

};

#endif /* SYMBOLMAP_HH */
