// mtrand.h
// C++ include file for MT19937, with initialization improved 2002/1/26.
// Coded by Takuji Nishimura and Makoto Matsumoto.
// Ported to C++ by Jasper Bedaux 2003/1/1 (see http://www.bedaux.net/mtrand/).
// Modified by Teemu Hirsimäki for personal use 2007/6/9
// - namespace, renaming, simplified, stdint.h, doxygen
// - forcing 32-bit to ensure identical results on 64-bit machines
// The generators (Jasper's versions) returning floating point numbers
// are based on a version by Isaku Wada, 2002/01/09
//
// Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura,
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// 3. The names of its contributors may not be used to endorse or promote
//    products derived from this software without specific prior written
//    permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Any feedback is very welcome.
// http://www.math.keio.ac.jp/matumoto/emt.html
// email: matumoto@math.keio.ac.jp
//
// Feedback about this version should be sent to Teemu Hirsimäki
// email: teemu.hirsimaki@iki.fi

#ifndef MTW_H
#define MTW_H

// Visual studio doesn't have stdint.h varjokal 24.3.2010
#ifdef _MSC_VER
#include <boost/cstdint.hpp>
using namespace boost;
#else
#include <stdint.h>
#endif

/** Namespace for Mersenne Twister random number generator. */
namespace mtw {

  /** Mersenne Twister random number generator.  Note that each
   * instance has its own state. */
  class Rnd { 
  public:

    /** Default constructor. */
    Rnd() 
    { 
      init();
      seed(5489UL); 
    }

    /** Constructor with 32-bit int as seed */
    Rnd(uint32_t s) 
    { 
      init();
      seed(s); 
    }

    /** Constructor with an array of 32-bit ints as seed. */
    Rnd(const uint32_t* array, int size) 
    { 
      init();
      seed(array, size); 
    }

    /** Set the seed as 32-bit integer. */
    void seed(uint32_t); 

    /** Set the seed as an array of 32-bit ints. */
    void seed(const uint32_t*, int size); 
    
    /** Generate 32-bit unsigned integer. */
    uint32_t u()
    { 
      if (p == n) 
	gen_state(); // new state vector needed
      // gen_state() is split off to be non-inline, because it is only
      // called once in every 624 calls and otherwise this would
      // become too big to get inlined
      uint32_t x = state[p++];
      x ^= (x >> 11);
      x ^= (x << 7) & 0x9D2C5680UL;
      x ^= (x << 15) & 0xEFC60000UL;
      return x ^ (x >> 18);
    }

    /** Generate float between [0,1). */ 
    float f() 
    {
      return static_cast<float>(u()) * (1. / 4294967296.); // 2^32
    } 

    /** Generate double between [0,1). */ 
    double d()
    {
      return static_cast<double>(u()) * (1. / 4294967296.); // 2^32
    } 

  private:

    static const int n = 624;
    static const int m = 397;

    /** State vector. */
    uint32_t state[n]; 

    /** Position in the state vector. */
    int p;

    /** Initialization used by constructors. */
    void init()
    {
      for (int i = 0; i < n; i++)
	state[i] = 0;
      p = 0;
    }

    /** Used by gen_state(). */
    uint32_t twiddle(uint32_t u, uint32_t v) 
    {
      return (((u & 0x80000000UL) | (v & 0x7FFFFFFFUL)) >> 1)
	^ ((v & 1UL) ? 0x9908B0DFUL : 0x0UL);
    }

    /** Generate a new state. */
    void gen_state();

    /** Copy constructor disabled. */
    Rnd(const Rnd&);

    /** Assignment disabled. */
    void operator=(const Rnd&);
  };

  /** Global generator instance for normal use. */
  extern Rnd rnd;

};

#endif
