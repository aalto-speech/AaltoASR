#include <string>
#include <stdio.h>
#include "Endian.hh"

int buf[3] = { 10, -1024, 12345678 };

int
main(int argc, char *argv[])
{
  std::string arg(argv[1]);

  if (!Endian::big)
    puts("this host is big");
  else
    puts("this host is little");

  if (arg == "w") {
    if (!Endian::big)
      Endian::convert_buffer(buf, 3, sizeof(int));

    fwrite(buf, sizeof(int), 3, stdout);
  }
  else if (arg == "r") {
    fread(buf, sizeof(int), 3, stdin);
    if (!Endian::big)
      Endian::convert_buffer(buf, 3, sizeof(int));

    for (int i = 0; i < 3; i++) {
      printf("%d\n", buf[i]);
    }
  }
}
