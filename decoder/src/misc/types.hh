#ifndef TYPES_HH
#define TYPES_HH

#include <vector>
#include <string>
#include "SymbolMap.hh"
#include <stdexcept>

typedef std::vector<bool> BoolVec;
typedef std::vector<int> IntVec;
typedef std::vector<float> FloatVec;
typedef std::vector<IntVec> IntVecVec;
typedef std::string Str;
typedef std::vector<Str> StrVec;
typedef misc::SymbolMap<Str,int> SymbolMap;
typedef std::exception Ex;
typedef std::runtime_error Error;

#endif /* TYPES_HH */
