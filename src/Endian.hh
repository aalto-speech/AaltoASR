#ifndef ENDIAN_HH
#define ENDIAN_HH

namespace Endian {
  extern bool big;
  
  void convert(void *buf, int bytes);
  void convert_buffer(void *buf, int elems, int bytes, int skip = 0);
};

#endif /* ENDIAN_HH */
