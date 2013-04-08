#include "mtw.hh"

namespace mtw {

  Rnd rnd;

  void Rnd::gen_state() 
  { 
    for (int i = 0; i < (n - m); ++i)
      state[i] = state[i + m] ^ twiddle(state[i], state[i + 1]);
    for (int i = n - m; i < (n - 1); ++i)
      state[i] = state[i + m - n] ^ twiddle(state[i], state[i + 1]);
    state[n - 1] = state[m - 1] ^ twiddle(state[n - 1], state[0]);
    p = 0;
  }

  void Rnd::seed(uint32_t s) 
  {
    state[0] = s & 0xFFFFFFFFU; 
    for (int i = 1; i < n; ++i) {
      state[i] = 1812433253U * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
      // see Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier in the
      // previous versions, MSBs of the seed affect only MSBs of the
      // array state 2002/01/09 modified by Makoto Matsumoto
      state[i] &= 0xFFFFFFFFU; 
    }
    p = n; // force gen_state() to be called for next random number
  }

  void Rnd::seed(const uint32_t* array, int size) 
  { 
    seed(19650218U);
    int i = 1, j = 0;
    for (int k = ((n > size) ? n : size); k; --k) {
      state[i] = (state[i] ^ ((state[i - 1] ^ (state[i - 1] >> 30)) * 
			      1664525U)) + array[j] + j;
      state[i] &= 0xFFFFFFFFU;
      ++j; j %= size;
      if ((++i) == n) { 
	state[0] = state[n - 1]; 
	i = 1; 
      }
    }
    for (int k = n - 1; k; --k) {
      state[i] = (state[i] ^ 
		  ((state[i - 1] ^ (state[i - 1] >> 30)) * 1566083941U)) - i;
      state[i] &= 0xFFFFFFFFU; 
      if ((++i) == n) { state[0] = state[n - 1]; i = 1; }
    }
    state[0] = 0x80000000U; // MSB is 1; assuring non-zero initial array
    p = n; // force gen_state() to be called for next random number
  }
  
};
