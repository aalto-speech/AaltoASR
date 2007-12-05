#include "Ziggurat.hh"


Ziggurat::Ziggurat()
{
  jsr=123456789;
  zigset(clock());
}


Ziggurat::~Ziggurat()
{
}


float
Ziggurat::nfix(void)
{
  r=3.442620f;
  for(;;)
  {
    x=hz*wn[iz];      /* iz==0, handles the base strip */
    if (iz==0)
    {
      do {
        x=-log(uni())*0.2904764; y=-log(uni());
      }	/* .2904764 is 1/r */
      while(y+y<x*x);
      return (hz>0)? r+x : -r-x;
    }
    /* iz>0, handle the wedges of other strips */
    if( fn[iz]+uni()*(fn[iz-1]-fn[iz]) < exp(-.5*x*x) ) return x;
    
    /* initiate, try to exit for(;;) for loop*/
    hz=shr3();
    iz=hz&127;
    if(fabs(hz)<kn[iz]) return (hz*wn[iz]);
  }
}


float
Ziggurat::efix(void)
{
  for(;;)
  {
    if(iz==0) return (7.69711-log(uni()));          /* iz==0 */
    x=jz*we[iz];
    if( fe[iz]+uni()*(fe[iz-1]-fe[iz]) < exp(-x) ) return (x);

    /* initiate, try to exit for(;;) loop */
    jz=shr3();
    iz=(jz&255);
    if(jz<ke[iz]) return (jz*we[iz]);
  }
}


void
Ziggurat::zigset(unsigned int jsrseed)
{
  const double m1 = 2147483648.0, m2 = 4294967296.;
  double dn=3.442619855899,tn=dn,vn=9.91256303526217e-3, q;
  double de=7.697117470131487, te=de, ve=3.949659822581572e-3;
  int i;
  jsr^=jsrseed;
  
  /* Set up tables for RNOR */
  q=vn/exp(-.5*dn*dn);
  kn[0]=(unsigned int)((dn/q)*m1);
  kn[1]=0;
  
  wn[0]=q/m1;
  wn[127]=dn/m1;
  
  fn[0]=1.;
  fn[127]=exp(-.5*dn*dn);
  
  for (i=126;i>=1;i--)
  {
    dn=sqrt(-2.*log(vn/dn+exp(-.5*dn*dn)));
    kn[i+1]=(unsigned int)((dn/tn)*m1);
    tn=dn;
    fn[i]=exp(-.5*dn*dn);
    wn[i]=dn/m1;
  }

  /* Set up tables for REXP */
  q = ve/exp(-de);
  ke[0]=(unsigned int)((de/q)*m2);
  ke[1]=0;
  
  we[0]=q/m2;
  we[255]=de/m2;
  
  fe[0]=1.;
  fe[255]=exp(-de);
  
  for (i=254;i>=1;i--)
  {
    de=-log(ve/de+exp(-de));
    ke[i+1]=(unsigned int)((de/te)*m2);
    te=de;
    fe[i]=exp(-de);
    we[i]=de/m2;
  }
}
