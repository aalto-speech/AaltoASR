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
    config("usage: gconvert [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('g', "gk=FILE", "arg must", "", "previous distributions (.gk)")
      ('o', "out=FILE", "arg must", "", "converted file (.gk)")
      ('d', "to-diagonal", "", "", "convert Gaussians to diagonal")
      ('f', "to-full", "", "", "convert Gaussians to full covariances")
      ('p', "to-pcgmm", "", "", "convert Gaussians to have a subspace constraint on precisions")
      ('s', "to-scgmm", "", "", "convert Gaussians to have an exponential subspace constraint")
      ('i', "info=INT", "arg", "0", "info level")
      ('\0', "ssdim=INT", "arg", "0", "subspace dimensionality")
      ('\0', "minvar=FLOAT", "arg", "0", "minimum diagonal variance")
      ;
    config.default_parse(argc, argv);

    int count=0;
    
    if (config["to-diagonal"].specified)
      count++;
    if (config["to-full"].specified)
      count++;
    if (config["to-pcgmm"].specified)
      count++;
    if (config["to-scgmm"].specified)
      count++;
    if (count==0)
      throw std::string("Define a target Gaussian type!");
    if (count>1)
      throw std::string("Define only one target Gaussian type!");
    
    PDFPool pool;
    PDFPool new_pool;
    pool.read_gk(config["gk"].get_str());
    new_pool.set_dim(pool.dim());

    
    // Initialize precision subspace if needed
    if (config["to-pcgmm"].specified) {
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
    }

    // Initialize exponential subspace if needed
    if (config["to-scgmm"].specified) {
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
      
      es->initialize_basis_pca(weights, covs, means, config["ssdim"].get_int());
    }
    
    Matrix covariance;
    Vector mean;
    for (int i=0; i<pool.size(); i++) {

      if (config["info"].get_int() > 0)
        std::cout << "Converting Gaussian: " << i << "/" << pool.size() << std::endl;
      
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
      if (config["to-pcgmm"].specified)
        new_gaussian = new PrecisionConstrainedGaussian(ps);
      if (config["to-scgmm"].specified)
        new_gaussian = new SubspaceConstrainedGaussian(es);
      
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
      
      new_gaussian->set_parameters(mean, covariance);

      new_pool.set_pdf(i, new_gaussian);
    }

    new_pool.write_gk(config["out"].get_str());
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  } 
}
