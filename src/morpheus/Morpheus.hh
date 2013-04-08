#ifndef MORPHEUS_HH
#define MORPHEUS_HH

#include <cstddef>  // NULL
#include "misc/macros.hh"
#include "misc/Trie.hh"
#include "fsalm/LM.hh"
#include "misc/ref.hh"
#include "misc/types.hh"

namespace mrf {

  using namespace fsalm;

  typedef std::map<Str,int> WordMap;

  class NoSeg : public std::exception { };

  class Morph {
  public:
    Morph() : sym(-1) { }
    Morph(int sym, Str str) : sym(sym), str(str) { }
    int sym;
    Str str;
  };

  typedef std::vector<Morph> MorphVec;

  class Path {
  public:

    Path() : reference_count(0) { };

    Path(Str morph, Path *path) 
      : morph(morph), path(path), reference_count(0) { }

    Str str() const 
    {
      std::vector<const Path*> stack;
      stack.push_back(this);
      while (stack.back()->path.ptr() != NULL)
        stack.push_back(stack.back()->path.ptr());
      if (stack.empty())
        return "";

      Str str;
      while (1) {
        str.append(stack.back()->morph);
        stack.pop_back();
        if (stack.empty())
          break;
        str.append(" ");
      }

      return str;
    }

    Str morph;
    ref::Ptr<Path> path;
    int reference_count;
  };

  class Token {
  public:
    Token() : pos(0), lm_node(0), score(0), reference_count(0) 
    { 
    }

    Token(const Token &t) { assert(false); }

    Token &operator=(const Token &t) { assert(false); }

    ~Token() {
    }

    void clone(const Token &t)
    {
      pos = t.pos;
      lm_node = t.lm_node;
      score = t.score;
      path = t.path;
    }

    int pos;
    int lm_node;
    float score;
    ref::Ptr<Path> path;
    int reference_count;
  };

  typedef std::vector<ref::Ptr<Token> > TokenPtrVec;
  typedef std::vector<TokenPtrVec> ActiveTokens;

  class Morpheus {
  public:

    Morpheus()
    {
      defaults();
    }

    Morpheus(LM *lm)
    {
      defaults();
      set_lm(lm);
    }

    void set_lm(LM *lm)
    {
      m_lm = lm;
      m_trie.clear();
      if (lm == NULL)
        return;
      
      // Build character trie for the morphs.
      //
      const SymbolMap &syms = lm->symbol_map();
      for (int s = 0; s < syms.size(); s++) {
        Str morph = syms.at(s);
        int n = m_trie.root();
        FOR(c, morph) {
          n = m_trie.insert(n, morph[c]);
        }
        m_trie.node(n).value = s;
      }

      reset();
    }

    void defaults()
    {
      set_lm(NULL);
      sentence_start_str = "<s>";
      sentence_end_str = "</s>";
      word_boundary_str = "<w>";
      unsegmented_char = "*";
      clear();
    }

    void clear()
    {
      m_active_tokens.clear();
    }

    void reset()
    {
      assert(m_lm != NULL);
      m_string = "";
      m_active_tokens.clear();
      m_active_tokens.resize(1);
      Token *token = new Token;
      token->lm_node = m_lm->initial_node_id();
      m_active_tokens[0].push_back(token);
    }

    void add_symbol(Str str)
    {
      TokenPtrVec tokens;
      tokens.swap(m_active_tokens.at(0));
      FOR(t, tokens) {
        Token *token = tokens[t];
        if (str != sentence_start_str) {
          int sym = m_lm->symbol_map().index(str);
          token->lm_node = m_lm->walk(token->lm_node, sym, &token->score);
        }
        token->path = new Path(str, token->path);
        p_activate_token(token);
      }
    }

    void add_string(Str str)
    {
      p_set_string(str);
      for (int i = 0; i < str.length(); i++) {
        p_process_pos(i);
      }

      if (m_active_tokens.back().empty())
        throw NoSeg();

      p_collapse_active_tokens();
    }

    Str str() 
    {
      assert(m_active_tokens.size() == 1);
      TokenPtrVec &vec = m_active_tokens.at(0);
      assert(vec.size() == 1);
      return vec.back()->path->str();
    }

    float score()
    {
      assert(m_active_tokens.size() == 1);
      TokenPtrVec &vec = m_active_tokens.at(0);
      assert(vec.size() == 1);
      return vec.back()->score;
    }

    Str sentence_start_str;
    Str sentence_end_str;
    Str word_boundary_str;
    Str unsegmented_char;

    WordMap unsegmented_words;
    int unsegmented_word_tokens;
    int unsegmented_word_types;

  private:

    void p_collapse_active_tokens()
    {
      std::swap(m_active_tokens.at(0), m_active_tokens.back());
      m_active_tokens.resize(1);
      TokenPtrVec &vec = m_active_tokens.at(0);
      FOR(t, vec) {
        vec[t]->pos = 0;
      }
    }

    void p_activate_token(Token *token)
    {
      assert(token->pos < m_active_tokens.size());
      TokenPtrVec &vec = m_active_tokens.at(token->pos);

      FOR(t, vec) {
        if (vec[t]->lm_node == token->lm_node) {
          if (vec[t]->score > token->score)
            return;
          vec[t].set(token);
          return;
        }
      }

      vec.push_back(token);
    }

    void p_propagate_token(Token *token, const MorphVec &morphs)
    {
      assert(token != NULL);
      FOR(i, morphs) {
        Token *new_token = new Token;

        // Ensure deletion if not referenced
        ref::Ptr<Token> dummy(new_token); 

        new_token->clone(*token);
        new_token->lm_node = m_lm->walk(new_token->lm_node, morphs[i].sym, 
                                        &new_token->score);
        new_token->pos += morphs[i].str.length();
        assert(new_token->pos > token->pos);
        new_token->path = new Path(morphs[i].str, new_token->path);
        p_activate_token(new_token);
      }
    }

    void p_generate_morphs(int pos, MorphVec &morphs)
    {
      morphs.clear();
      int n = m_trie.root();
      Str morph_str;
      for (int p = pos; p <= m_string.length(); p++) {
        
        n = m_trie.find(n, m_string[p]);
        if (n < 0)
          return;
        morph_str.push_back(m_string[p]);

        int sym = m_trie.node(n).value;
        if (sym < 0)
          continue;

        morphs.push_back(Morph(sym, morph_str));
      }
    }

    void p_process_pos(int pos)
    {
      assert(pos < m_string.length());
      if (m_active_tokens[pos].empty())
        return;
      MorphVec morphs;
      p_generate_morphs(pos, morphs);
      FOR(t, m_active_tokens[pos])
        p_propagate_token(m_active_tokens[pos][t], morphs);
      m_active_tokens[pos].clear();
    }

    void p_set_string(Str str)
    {
      assert(str.length() > 0);
      assert(m_active_tokens.size() == 1);
      m_string = str;
      m_active_tokens.resize(m_string.length() + 1);
    }

    /** Language model used in segmentation. */
    LM *m_lm;

    /** Character trie mapping the morphs to morph indices. */
    misc::Trie<unsigned char, int, -1> m_trie;

    /** Active tokens for each position in the sentence. */
    std::vector<TokenPtrVec> m_active_tokens;

    /** Current string to be segmented. */
    Str m_string;
  
  };

};

#endif /* MORPHEUS_HH */
