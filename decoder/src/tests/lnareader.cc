#include <iomanip>
#include <iostream>

#include "LnaReaderCircular.hh"
#include "LnaReader.hh"

int num_models = 76;
LnaReaderCircular lna;

void
test(int frame, Acoustics &acu)
{
  std::cout << std::setw(8) << frame;
  try {
    bool ok = acu.go_to(frame);

    if (!ok)
      std::cout << "eof" << std::endl;
    else {
      std::cout << std::setw(8) << acu.log_prob(0)
		<< std::setw(8) << acu.log_prob(num_models - 1)
		<< std::endl;
    }
  }
  catch (LnaReaderCircular::FrameForgotten &e) {
    std::cout << "forgotten" << std::endl;
  }
}

int
main(int argc, char *argv[])
{
  std::cout.setf(std::cout.fixed, std::cout.floatfield);
  std::cout.setf(std::cout.left, std::cout.adjustfield);
  std::cout.precision(2);

  lna.open("/home/neuro/thirsima/share/synt/pk_synt5.lna", num_models, 1024);

  test(0,lna);
  test(1,lna);
  test(2,lna);

  test(1023,lna);
  test(0,lna);
  test(1,lna);
  test(2,lna);

  test(1023,lna);
  test(1024,lna);
  test(0,lna);
  test(1,lna);
  test(2,lna);

  test(1023,lna);
  test(1024,lna);
}
