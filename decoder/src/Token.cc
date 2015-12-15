#include <vector>
#include <sstream>
#include <algorithm>  // reverse

#include "Token.hh"

using namespace std;

void Token::get_state_history(vector<StateHistory *> & result) const
{
  result.clear();
  for (Token::StateHistory * ptr = state_history;
       ptr != nullptr && ptr->previous != nullptr;
       ptr = ptr->previous) {
    result.push_back(ptr);
  }
  reverse(result.begin(), result.end());
}

void Token::get_lm_history(vector<LMHistory *> & result,
                           LMHistory * limit) const
{
  result.clear();
  for (LMHistory * ptr = lm_history;
       ptr != limit && ptr->last().word_id() >= 0;
       ptr = ptr->previous) {
    result.push_back(ptr);
  }
  reverse(result.begin(), result.end());
}
