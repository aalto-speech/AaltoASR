#ifndef SEARCH_HH
#define SEARCH_HH

#include "misc/macros.hh"
#include "misc/ref.hh"
#include "misc/types.hh"
#include "fsalm/LM.hh"

namespace mrf {

  class NoSeg : public std::exception {
  };

  class Morph {
  public:
    Morph() : sym(-1), len(-1) { }
    Morph(int sym, int len) : sym(sym), len(len) { }
    int sym;
    int len;
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
      previous = t.previous;
    }

    int pos;
    int lm_node;
    float score;
    ref::Ptr<Token> previous;
    int reference_count;
  };

  typedef std::vector<ref::Ptr<Token> > TokenPtrVec;

  class Search {
  public:
    Search() :
      lm(NULL) {}

    Search(fsalm::LM *lm_)
      : lm(lm_) { }

    fsalm::LM *lm;
    Str word;
    std::vector<TokenPtrVec> active_tokens;
    std::vector<Morph> morphs;

    /** Segment word into morphs.
     * \param word_ = word to segment
     * \return string containing white-space separated morphs
     * \throw NoSeg() if no segmentation possible
     */
    Str segment_word(Str word_) 
    {
      reset(word_);
      for (int i = 0; i < word_.length(); i++) 
        process_pos(i);
      TokenPtrVec &vec = active_tokens.at(word_.length());
      if (vec.size() != 1)
        throw std::runtime_error(
          "ERROR: something went wrong, several tokens survived for word: "
          + word_);
      return str(vec.back());
    }

    void reset(Str word_)
    {
      assert(word_.length() > 0);
      word = word_;
      active_tokens.clear();
      active_tokens.resize(word.length() + 1);
      Token *token = new Token;
      token->lm_node = lm->initial_node_id();
      active_tokens[0].push_back(token);
    }

    void activate_token(Token *token)
    {
      assert(token->pos <= word.length());
      assert(token->previous.ptr() != token);
      TokenPtrVec &vec = active_tokens.at(token->pos);

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

    void propagate_token(Token *token) 
    {
      assert(token != NULL);
      FOR(i, morphs) {
        Token *new_token = new Token;

        // Ensure deletion if not referenced
        ref::Ptr<Token> dummy(new_token); 

        new_token->clone(*token);
        new_token->lm_node = lm->walk(new_token->lm_node, morphs[i].sym, 
                                      &new_token->score);
        new_token->pos += morphs[i].len;
        assert(new_token->pos > token->pos);
        new_token->previous.set(token);
        if (new_token->pos == word.length())
          new_token->lm_node = lm->walk(new_token->lm_node, lm->end_symbol(),
                                        &new_token->score);
        activate_token(new_token);
      }
    }

    void generate_morphs(int pos)
    {
      morphs.clear();
      for (int i = 1; i <= word.length() - pos; i++) {
        Str morph = word.substr(pos, i);
        int sym = lm->symbol_map().index_nothrow(morph);
        if (sym < 0)
          continue;
        morphs.push_back(Morph(sym, morph.length()));
      }
      if (morphs.empty())
        throw NoSeg();
    }

    void process_pos(int pos)
    {
      assert(pos < word.length());
      if (active_tokens[pos].empty())
        return;
      generate_morphs(pos);
      FOR(t, active_tokens[pos]) {
        propagate_token(active_tokens[pos][t]);
      }
      active_tokens[pos].clear();
    }

    Str str(Token *token) 
    {
      Str s;
      IntVec pos;
      while (token != NULL) {
        pos.push_back(token->pos);
        token = token->previous;
      }
      while (pos.size() > 1) {
        int start = pos.back();
        pos.pop_back();
        s.append(word.substr(start, pos.back() - start));
        if (pos.size() > 1)
          s.append(" ");
      }

      return s;
    }

    void debug_print()
    {
      FOR(i, active_tokens) {
        fprintf(stderr, "%zd: %zd tokens\n", i, active_tokens[i].size());
      }
    }

  };

};

#endif /* SEARCH_HH */
