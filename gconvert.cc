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
      ('d', "to-diagonal", "", "", "convert Gaussians to diagonal")
      ('f', "to-full", "", "", "convert Gaussians to full covariances")
      ('\0', "minvar=FLOAT", "arg", "0", "minimum diagonal variance")
      ;
    config.default_parse(argc, argv);

    if (config["to-diagonal"].specified && config["to-full"].specified)
      throw std::string("Don't define both -d and -f!");

    if (!config["to-diagonal"].specified && !config["to-full"].specified)
      throw std::string("Define either -d and -f!");

    PDFPool pool;
    PDFPool new_pool;
    pool.read_gk(config["gk"].get_str());
    new_pool.set_dim(pool.dim());
    
    Matrix covariance;
    Vector mean;
    for (int i=0; i<pool.size(); i++) {

      // Fetch source Gaussian
      Gaussian *gaussian = dynamic_cast< Gaussian* > (pool.get_pdf(i));
      if (gaussian == NULL)
        continue;

      // Create target Gaussian
      Gaussian *new_gaussian = NULL;
      if (config["to-diagonal"].specified)
        new_gaussian = new DiagonalGaussian(gaussian->dim());
      if (config["to-full"].specified)
        new_gaussian = new FullCovarianceGaussian(gaussian->dim());
      
      // Convert
      gaussian->get_covariance(covariance);
      gaussian->get_mean(mean);

      if (config["minvar"].specified)
      {
        double minvar = config["minvar"].get_float();
        for (int j = 0; j < gaussian->dim(); j++)
          if (covariance(j,j) < minvar)
            covariance(j,j) = minvar;
      }
      
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
