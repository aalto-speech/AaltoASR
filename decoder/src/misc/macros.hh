#ifndef MACROS_HH
#define MACROS_HH

#define FOR(i,vec) for (size_t i = 0; i < vec.size(); i++)

#ifdef USE_SANITY
# define SANITY if (1)
#else
# define SANITY if (0)
#endif

#ifdef USE_DEBUG
# define DEBUG if (1)
#else
# define DEBUG if (0)
#endif

#endif /* MACROS_HH */
