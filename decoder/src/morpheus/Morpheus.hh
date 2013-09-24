#ifndef MORPHEUS_HH
#define MORPHEUS_HH

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "misc/Trie.hh"
#include "misc/util.hh"
#include "fsalm/LM.hh"

using namespace std;
using namespace fsalm;


namespace mrf {


class NoSeg : public exception { };

class Morph {
public:
    Morph() : sym(-1) { }
    Morph(int sym, string str) : sym(sym), str(str) { }
    int sym;
    string str;
};


class Path {
public:

    Path() { };

    Path(string morph, shared_ptr<Path> path)
        : morph(morph), path(path) { }

    string str() const
    {
        vector<const Path*> stack;
        stack.push_back(this);
        while (stack.back()->path != NULL)
            stack.push_back(stack.back()->path.get());
        if (stack.empty())
            return "";

        string str;
        while (1) {
            str.append(stack.back()->morph);
            stack.pop_back();
            if (stack.empty())
                break;
            str.append(" ");
        }

        return str;
    }

    string morph;
    shared_ptr<Path> path;
};

class Token {
public:
    Token() : pos(0), lm_node(0), score(0), soft_score(0)
    {
    }

    Token(const Token &t) {
        assert(false);
    }

    Token &operator=(const Token &t) {
        assert(false);
    }

    ~Token() {
    }

    void clone(const Token &t)
    {
        pos = t.pos;
        lm_node = t.lm_node;
        score = t.score;
        soft_score = t.soft_score;
        path = t.path;
    }

    int pos;
    int lm_node;
    float score;
    float soft_score;
    shared_ptr<Path> path;
};


typedef vector<shared_ptr<Token> > TokenPtrVec;


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
        const SymbolMap &syms = lm->symbol_map();
        for (int s = 0; s < syms.size(); s++) {
            string morph = syms.at(s);
            int n = m_trie.root();
            for (int i=0; i<morph.size(); i++) n = m_trie.insert(n, morph[i]);
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
        merge_same_lm_state = true;
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
        auto token = make_shared<Token>();
        token->lm_node = m_lm->initial_node_id();
        m_active_tokens[0].push_back(token);
    }

    void add_symbol(string str, bool cumulate_score=true)
    {
        TokenPtrVec tokens;
        tokens.swap(m_active_tokens.at(0));

        for (auto token = tokens.begin(); token != tokens.end(); ++token) {
            if (str != sentence_start_str) {
                int sym = m_lm->symbol_map().index(str);
                if (cumulate_score) {
                    float curr_prob = 0.0;
                    (*token)->lm_node = m_lm->walk((*token)->lm_node, sym, &curr_prob);
                    (*token)->score += curr_prob;
                    (*token)->soft_score += curr_prob;
                }
                else
                    (*token)->lm_node = m_lm->walk((*token)->lm_node, sym, NULL);
            }
            (*token)->path = make_shared<Path>(str, (*token)->path);
            p_activate_token(*token);
        }
    }

    void add_string(string str)
    {
        p_set_string(str);
        for (int i = 0; i < str.length(); i++)
            p_process_pos(i);

        if (m_active_tokens.back().empty())
            throw NoSeg();

        p_collapse_active_tokens();
    }

    string str()
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

    float soft_score()
    {
        assert(m_active_tokens.size() == 1);
        TokenPtrVec &vec = m_active_tokens.at(0);
        assert(vec.size() == 1);
        return vec.back()->soft_score;
    }

    string sentence_start_str;
    string sentence_end_str;
    string word_boundary_str;
    bool merge_same_lm_state;


private:

    void p_collapse_active_tokens()
    {
        swap(m_active_tokens.at(0), m_active_tokens.back());
        m_active_tokens.resize(1);
        TokenPtrVec &vec = m_active_tokens.at(0);
        for (auto token = vec.begin(); token != vec.end(); ++token)
            (*token)->pos = 0;
    }

    void p_activate_token(shared_ptr<Token> token)
    {
        assert(token->pos < m_active_tokens.size());
        TokenPtrVec &vec = m_active_tokens.at(token->pos);

        if (merge_same_lm_state) {
            for (auto tokenit = vec.begin(); tokenit != vec.end(); ++tokenit) {
                if ((*tokenit)->lm_node == token->lm_node) {
                    float soft_score = util::log10addf(token->soft_score, (*tokenit)->soft_score);
                    token->soft_score = soft_score;
                    (*tokenit)->soft_score = soft_score;
                    if ((*tokenit)->score > token->score)
                        return;
                    (*tokenit) = token;
                    return;
                }
            }
        }

        vec.push_back(token);
    }

    void p_propagate_token(shared_ptr<Token> token, const vector<Morph> &morphs)
    {
        assert(token != NULL);

        for (auto morph = morphs.begin(); morph != morphs.end(); ++morph) {
            auto new_token = make_shared<Token>();
            new_token->clone(*token);
            float curr_prob = 0.0;
            new_token->lm_node = m_lm->walk(new_token->lm_node, (*morph).sym, &curr_prob);
            new_token->score += curr_prob;
            new_token->soft_score += curr_prob;
            new_token->pos += (*morph).str.length();
            assert(new_token->pos > token->pos);
            new_token->path = make_shared<Path>((*morph).str, new_token->path);
            p_activate_token(new_token);
        }
    }

    void p_generate_morphs(int pos, vector<Morph> &morphs)
    {
        morphs.clear();
        int n = m_trie.root();
        string morph_str;
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
        vector<Morph> morphs;
        p_generate_morphs(pos, morphs);
        for (auto tokenit = m_active_tokens[pos].begin(); tokenit != m_active_tokens[pos].end(); ++tokenit)
            p_propagate_token(*tokenit, morphs);
        m_active_tokens[pos].clear();
    }

    void p_set_string(string str)
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
    vector<TokenPtrVec> m_active_tokens;

    /** Current string to be segmented. */
    string m_string;

};

};

#endif /* MORPHEUS_HH */
