#include <iomanip>
#include <iostream>

#include "LnaReaderCircular.hh"

int num_models = 76;
LnaReaderCircular lna;

void
test(int frame)
{
  std::cout << std::setw(8) << frame;
  try {
    bool ok = lna.go_to(frame);

    if (!ok)
      std::cout << "eof" << std::endl;
    else {
      std::cout << std::setw(8) << lna.log_prob(0)
		<< std::setw(8) << lna.log_prob(num_models - 1)
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

  test(0);
  test(1023);
  test(0);
  test(1024);
  test(0);
}
