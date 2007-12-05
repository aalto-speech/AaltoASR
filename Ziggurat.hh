#ifndef ZIGGURAT_HH
#define ZIGGURAT_HH

/*
  The ziggurat method for RNOR and REXP
  Combine the code below with the main program in which you want
  normal or exponential variates.   Then use of RNOR in any expression
  will provide a standard normal variate with mean zero, variance 1,
  while use of REXP in any expression will provide an exponential variate
  with density exp(-x),x>0.
  Before using RNOR or REXP in your main, insert a command such as
  zigset(86947731 );
  with your own choice of seed value>0, rather than 86947731.
  (If you do not invoke zigset(...) you will get all zeros for RNOR and REXP.)
  For details of the method, see Marsaglia and Tsang, "The ziggurat method
  for generating random variables", Journ. Statistical Software.
  
  Original functions converted to a c++ object and static variables removed
  // mavarjok
*/

#include <cmath>
#include <ctime>

class Ziggurat {

public:
  
  Ziggurat();
  ~Ziggurat();

  /* Generates a random sample from N(0,1) distribution */
  inline float rnor();
  /* Generates a random sample from an exponential distribution */
  inline float rexp();
  
private:
  
  unsigned int jz,jsr;
  int hz;
  unsigned int iz, kn[128], ke[256];
  float wn[128],fn[128], we[256],fe[256];
  float x, y;
  float r;     /* The start of the right tail */

  /* Generates variates from the residue when rejection in RNOR occurs */
  float nfix(void);

  /* Generates variates from the residue when rejection in REXP occurs */
  float efix(void);

  /* Sets the seed and creates the tables */
  void zigset(unsigned int jsrseed);

  inline unsigned int shr3();
  inline double uni();
  inline unsigned int iuni();
};


unsigned int
Ziggurat::shr3()
{
  jz=jsr;
  jsr^=(jsr<<13);
  jsr^=(jsr>>17);
  jsr^=(jsr<<5);
  return jz+jsr;
}


double
Ziggurat::uni()
{
  return .5 + (signed) shr3()*.2328306e-9;
}


unsigned int
Ziggurat::iuni()
{
  return shr3();
}


float
Ziggurat::rnor()
{
  hz=shr3();
  iz=hz&127;
  return ((fabs(hz)<kn[iz]) ? hz*wn[iz] : nfix());
}


float
Ziggurat::rexp()
{
  jz=shr3();
  iz=jz&255;
  return ((jz <ke[iz]) ? jz*we[iz] : efix());
}


#endif /* ZIGGURAT_HH */
