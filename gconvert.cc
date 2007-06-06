#include "Distributions.hh"
#include "conf.hh"
#include "LinearAlgebra.hh"

conf::Config config;

int
main(int argc, char *argv[])
{
  try {
    config("usage: gconvert [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('g', "gk=FILE", "arg must", "", "previous distributions (.gk)")
      ('o', "out=FILE", "arg must", "", "converted file (.gk)")
      ('\0', "2d", "", "", "convert Gaussians to diagonal")
      ('\0', "2f", "", "", "convert Gaussians to full covariances")
      ;
    config.default_parse(argc, argv);

    if (config["2d"].specified && config["2f"].specified)
      throw std::string("Don't define both --2d and --2f!");

    if (!config["2d"].specified && !config["2f"].specified)
      throw std::string("Define either --2d and --2f!");

    PDFPool pool;
    PDFPool new_pool;
    pool.read_gk(config["gk"].get_str());

    Matrix covariance;
    Vector mean;
    for (int i=0; i<pool.size(); i++) {

      // Fetch source Gaussian
      Gaussian *gaussian = dynamic_cast< Gaussian* > (pool.get_pdf(i));
      if (gaussian == NULL)
        continue;

      // Create target Gaussian
      Gaussian *new_gaussian = NULL;
      if (config["2d"].specified)
        new_gaussian = new DiagonalGaussian(gaussian->dim());
      if (config["2f"].specified)
        new_gaussian = new FullCovarianceGaussian(gaussian->dim());
      
      // Convert
      gaussian->get_covariance(covariance);
      gaussian->get_mean(mean);
      new_gaussian->set_covariance(covariance);
      new_gaussian->set_mean(mean);

      new_pool.set_pdf(i, new_gaussian);
    }

    new_pool.write_gk(config["out"].get_str());
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  } 
}
