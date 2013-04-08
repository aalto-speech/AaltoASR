#ifndef ENDIAN_HH
#define ENDIAN_HH

/// Tools for handling conversions between different byte orders.
namespace Endian {
  /// Is this host big-endian
  extern bool big;
  
  /// Convert the endianity of single value
  void convert(void *buf, int bytes);

  /// Convert the endianity of 'elems' elements of 'bytes' bytes.
  void convert_buffer(void *buf, int elems, int bytes, int skip = 0);
};

#endif /* ENDIAN_HH */
