#include "Distributions.hh"
#include "conf.hh"
#include "LinearAlgebra.hh"

conf::Config config;
PrecisionSubspace *ps;
ExponentialSubspace *es;


int
main(int argc, char *argv[])
{
  try {
    config("usage: subspace [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('g', "gk=FILE", "arg must", "", "Gaussian distributions (.gk)")
      ('o', "out=FILE", "arg must", "", "output file for the subspace")
      ('p', "pcgmm", "", "", "initialize a precision subspace")
      ('s', "scgmm", "", "", "initialize an exponential subspace")
      ('i', "info=INT", "arg", "0", "info level")
      ('d', "ssdim=INT", "arg", "0", "subspace dimensionality")
      ;
    config.default_parse(argc, argv);

    if (config["pcgmm"].specified && config["scgmm"].specified)
      throw std::string("Define only one subspace type!");
    
    PDFPool pool;
    pool.read_gk(config["gk"].get_str());

    // Initialize precision subspace
    if (config["pcgmm"].specified) {
      if (config["ssdim"].get_int() <= 0)
        throw std::string("The subspace dimensionality must be above zero!");
      ps = new PrecisionSubspace(config["ssdim"].get_int(), pool.dim());

      if (config["info"].get_int() > 0)
        std::cout << "Initializing the precision subspace\n";
        
      std::vector<double> weights;
      std::vector<LaGenMatDouble> covs;
      weights.resize(pool.size());
      covs.resize(pool.size());
      
      for (int i=0; i<pool.size(); i++) {        
        // Fetch source Gaussian
        Gaussian *gaussian = dynamic_cast< Gaussian* > (pool.get_pdf(i));
        if (gaussian == NULL)
          continue;
        gaussian->get_covariance(covs[i]);
        weights[i] = 1;
      }

      ps->initialize_basis_pca(weights, covs, config["ssdim"].get_int());

      // Write out
      std::ofstream out(config["out"].get_str().c_str());
      ps->write_subspace(out);
      out.close();
    }

    // Initialize exponential subspace if needed
    if (config["scgmm"].specified) {
      if (config["ssdim"].get_int() <= 0)
        throw std::string("The subspace dimensionality must be above zero!");
      es = new ExponentialSubspace(config["ssdim"].get_int(), pool.dim());

      if (config["info"].get_int() > 0)
        std::cout << "Initializing the exponential subspace\n";
      
      std::vector<double> weights;
      std::vector<LaVectorDouble> means;
      std::vector<LaGenMatDouble> covs;
      weights.resize(pool.size());
      covs.resize(pool.size());
      means.resize(pool.size());
      
      for (int i=0; i<pool.size(); i++) {        
        // Fetch source Gaussian
        Gaussian *gaussian = dynamic_cast< Gaussian* > (pool.get_pdf(i));
        if (gaussian == NULL)
          continue;
        gaussian->get_covariance(covs[i]);
        gaussian->get_mean(means[i]);
        weights[i] = 1;
      }

      // Initialize the subspace
      es->initialize_basis_pca(weights, covs, means, config["ssdim"].get_int());

      // Write out
      std::ofstream out(config["out"].get_str().c_str());
      es->write_subspace(out);
      out.close();
    }
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  } 
}
