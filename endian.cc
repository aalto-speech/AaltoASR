#include "endian.hh"

namespace endian {
  int BIG_ENDIAN_TEST = 0x12;
  bool big = (*((unsigned char *)&BIG_ENDIAN_TEST) != 0x12);

  void convert(void *buf, int bytes)
  {
    unsigned char *ptr = (unsigned char*)buf;
    unsigned char tmp;
    for (int i = 0; i < bytes / 2; i++) {
      tmp = ptr[i];
      ptr[i] = ptr[bytes - i - 1];
      ptr[bytes - i - 1] = tmp;
    }
  }

  void convert_buffer(void *buf, int elems, int bytes, int skip)
  {
    unsigned char *ptr = (unsigned char*)buf;
    unsigned char tmp;
    while (elems > 0) {
      for (int i = 0; i < bytes / 2; i++) {
	tmp = ptr[i];
	ptr[i] = ptr[bytes - i - 1];
	ptr[bytes - i - 1] = tmp;
      }
      ptr += (bytes + skip);
      elems--;
    }
  }

};
