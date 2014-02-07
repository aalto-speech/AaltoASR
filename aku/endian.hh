#ifndef ENDIAN_HH
#define ENDIAN_HH

#include <stdio.h>
#include "assert.h"

/// Tools for handling conversions between different byte orders.
namespace endian {
  /// Is this host big-endian
  extern bool big;
  
  /// Convert the endianity of single value
  void convert(void *buf, int bytes);

  /// Convert the endianity of 'elems' elements of 'bytes' bytes.
  void convert_buffer(void *buf, int elems, int bytes, int skip = 0);

  /// Parse 4-byte value in little-endian format
  template <typename T>
  T get4(void *buf, bool read_big = false)
  {
    assert(sizeof(T) == 4);
    if (big == read_big)
      return *(T*)buf;

    T ret = 0;
    char *ptr = (char*)&ret;
    char *src = (char*)buf;
    ptr[0] = src[3];
    ptr[1] = src[2];
    ptr[2] = src[1];
    ptr[3] = src[0];
    return ret;
  }

  /// Write 4-byte value in little-endian format
  template <typename T>
  void put4(T value, void *buf, bool write_big = false)
  {
    assert(sizeof(T) == 4);
    if (big == write_big) {
      *(T*)buf = value;
      return;
    }

    char *ptr = (char*)&value;
    char *tgt = (char*)buf;
    tgt[0] = ptr[3];
    tgt[1] = ptr[2];
    tgt[2] = ptr[1];
    tgt[3] = ptr[0];
  }

  /// Write 4-byte value in little-endian format
  template <typename T>
  size_t write4(T v, FILE *file, bool write_big = false)
  {
    assert(sizeof(v) == 4);
    if (big == write_big)
      return fwrite(&v, 4, 1, file);

    char *ptr = (char*)(&v);
    char little[4];
    little[0] = ptr[3];
    little[1] = ptr[2];
    little[2] = ptr[1];
    little[3] = ptr[0];
    return fwrite(little, 4, 1, file);
  }

  /// Read 4-byte value in little-endian format
  template <typename T>
  size_t read4(T *buf, FILE *file, bool read_big = false)
  {
    if (big == read_big)
      return fread(buf, 4, 1, file);

    char *ptr = (char*)buf;
    char little[4];
    size_t ret = fread(little, 4, 1, file);
    ptr[0] = little[3];
    ptr[1] = little[2];
    ptr[2] = little[1];
    ptr[3] = little[0];
    return ret;
  }

  /// Write 2-byte value in little-endian format
  template <typename T>
  size_t write2(T v, FILE *file, bool write_big = false)
  {
    assert(sizeof(v) == 2);
    if (big == write_big)
      return fwrite(&v, 2, 1, file);

    char *ptr = (char*)(&v);
    char little[4];
    little[0] = ptr[0];
    little[1] = ptr[1];
    return fwrite(little, 2, 1, file);
  }

  /// Read 2-byte value in little-endian format
  template <typename T>
  size_t read2(T *buf, FILE *file, bool read_big = false)
  {
    if (big == read_big)
      return fread(buf, 2, 1, file);

    char *ptr = (char*)buf;
    char little[4];
    size_t ret = fread(little, 2, 1, file);
    ptr[0] = little[0];
    ptr[1] = little[1];
    return ret;
  }

};

#endif /* ENDIAN_HH */
