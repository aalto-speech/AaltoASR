#ifndef LEXICON_HH
#define LEXICON_HH

#include <cstddef>  // NULL
#include <vector>
#include <deque>
#include <assert.h>

#include "Lexicon.hh"

/* 

   Notes about the implementation:

   - Currently Expander keeps a vector of Token pointers instead of
   classes.  This is more efficient if the sizeof(Token) is big enough
   (above 24 bytes is probably ok). FIXME

*/



// FIXME: Lexicon nodes must have costs for different forms of a word.

class Lexicon {
public:

  // NOTE: It seems that we have to use token pointers in
  // Lexicon::State and Expander.  Otherwise it is impossible to point
  // from Lexicon::State to the corresponding token vector in
  // Expander, because Expander alters the order of tokens.

  class Token;
  
  class State {
  public:
    inline State() : incoming_token(0), outgoing_token(0) { }
    Token *incoming_token;
    Token *outgoing_token;
  };

  class Node {
  public:
    // Note that NowayLexiconReader assumes that word_id's are
    // negative by default.
    inline Node() : log_prob(0), word_id(-1), hmm_id(-1) { }
    inline Node(int states) : 
      log_prob(0), word_id(-1), hmm_id(-1), states(states) { }
    inline Node(const Node &node) :
      log_prob(node.log_prob), word_id(node.word_id), hmm_id(node.hmm_id),
      states(node.states.size()) { }
    float log_prob;
    int word_id;
    int hmm_id;
    std::vector<State> states;
    std::vector<Node*> next;
  };

  class Path {
  public:
    inline Path(int hmm_id, int frame, float log_prob, class Path *prev);
    inline ~Path();
    inline void link() { m_reference_count++; }
    inline static void unlink(Path *path);
    static int count;

    int hmm_id;
    int frame;
    float log_prob;
    class Path *prev;
  private:
    int m_reference_count;
  };

  // FIXME: if we do not need the path, we can remove reference counting
  // and special copying functions.
  class Token {
  public:
    inline Token();
    inline Token(const Token &t);
    inline Token &operator=(const Token &token);
    inline ~Token();
    inline void add_path(int hmm_id, int frame, float log_prob);
    inline void reset();

    int frame; // FIXME: do we need frame counter in token?
    unsigned char state_duration;
    char state;
    Lexicon::Node *node;
    float log_prob;
    float dur_log_prob;
    Path *path;
  };

  Lexicon();
  Lexicon(const Lexicon &lexicon) { assert(false); } // never copy lexicon
  inline Lexicon::Node *root() { return &m_root_node; }
  inline int words() const { return m_words; }
  inline void update_words(int word) 
    { 
      if (word >= m_words)
	m_words = word + 1;
    }

private:
  int m_words; // Largest word_id in the nodes plus one
  Node m_root_node;
};

//////////////////////////////////////////////////////////////////////

Lexicon::Path::Path(int hmm_id, int frame, float log_prob, Path *prev)
  : hmm_id(hmm_id),
    frame(frame),
    log_prob(log_prob),
    prev(prev),
    m_reference_count(0)
{
  count++;
  if (prev)
    prev->link();
}

Lexicon::Path::~Path()
{
  count--;
}

void
Lexicon::Path::unlink(Path *path)
{
  while (path->m_reference_count == 1) {
    Path *prev = path->prev;
    delete path;
    path = prev;
    if (!path)
      return;
  }
  path->m_reference_count--;
}

Lexicon::Token::Token()
  : frame(-1),
    state_duration(0),
    state(0),
    node(NULL),
    log_prob(0),
    dur_log_prob(0),
    path(NULL)
{
}

Lexicon::Token::Token(const Token &t)
  : frame(t.frame),
    state_duration(t.state_duration),
    state(t.state),
    node(t.node),
    log_prob(t.log_prob),
    dur_log_prob(t.log_prob),
    path(t.path)
{
  if (path)
    path->link();
}

Lexicon::Token&
Lexicon::Token::operator=(const Token &t)
{
  if (this == &t)
    return *this;

  Path *old_path = path;

  frame = t.frame;
  state_duration = t.state_duration;
  state = t.state;
  node = t.node;
  log_prob = t.log_prob;
  dur_log_prob = t.dur_log_prob;
  path = t.path;
  if (path)
    path->link();

  if (old_path)
    Path::unlink(old_path);

  return *this;
}

Lexicon::Token::~Token()
{
  if (path)
    Path::unlink(path);
}

void
Lexicon::Token::add_path(int hmm_id, int frame, float log_prob)
{
  Path *old_path = path;
  path = new Path(hmm_id, frame, log_prob, path);
  path->link();
  if (old_path)
    Path::unlink(old_path);
}

void
Lexicon::Token::reset()
{
  if (path)
    Path::unlink(path);
  frame = -1;
  state_duration = 0;
  state = 0;
  node = NULL;
  log_prob = 0;
  dur_log_prob = 0;
  path = NULL;
}

#endif /* LEXICON_HH */
