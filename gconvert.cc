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
      ('C', "coeffs=NAME", "arg", "", "Precomputed precision/subspace coefficients")
      ('d', "to-diagonal", "", "", "convert Gaussians to diagonal")
      ('f', "to-full", "", "", "convert Gaussians to full covariances")
      ('p', "to-pcgmm", "", "", "convert Gaussians to have a subspace constraint on precisions")
      ('s', "to-scgmm", "", "", "convert Gaussians to have an exponential subspace constraint")
      ('b', "subspace=FILE", "arg", "", "use an already initialized subspace")
      ('i', "info=INT", "arg", "0", "info level")
      ('\0', "ssdim=INT", "arg", "0", "subspace dimensionality")
      ('\0', "minvar=FLOAT", "arg", "0", "minimum diagonal variance")
      ('\0', "hcl_bfgs_cfg=FILE", "arg", "", "configuration file for HCL biconjugate gradient algorithm")
      ('\0', "hcl_line_cfg=FILE", "arg", "", "configuration file for HCL line search algorithm")
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

    // Linesearch for subspace models
    HCL_LineSearch_MT_d ls;
    if (config["hcl_line_cfg"].specified)
      ls.Parameters().Merge(config["hcl_line_cfg"].get_str().c_str());
    
    // lmBFGS for subspace models
    HCL_UMin_lbfgs_d bfgs(&ls);
    if (config["hcl_bfgs_cfg"].specified)
      bfgs.Parameters().Merge(config["hcl_bfgs_cfg"].get_str().c_str());
    
    
    // Initialize the Pcgmm case
    if (config["to-pcgmm"].specified) {

      ps = new PrecisionSubspace(config["ssdim"].get_int(), pool.dim());
      
      // Read from file
      if (config["subspace"].specified) {
        std::ifstream in(config["subspace"].get_str().c_str());
        ps->read_subspace(in);
        in.close();
      }
      // Or initialize
      else {
        if (config["ssdim"].get_int() <= 0)
          throw std::string("The subspace dimensionality must be above zero!");
        
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

      ps->set_hcl_optimization(&ls, &bfgs, config["hcl_line_cfg"].get_str(), config["hcl_bfgs_cfg"].get_str());
      new_pool.set_precision_subspace(1, ps);
    }
    
    // Initialize the Scgmm case
    if (config["to-scgmm"].specified) {

      es = new ExponentialSubspace(config["ssdim"].get_int(), pool.dim());

      // Read from file
      if (config["subspace"].specified) {
        std::ifstream in(config["subspace"].get_str().c_str());
        es->read_subspace(in);
        in.close();
      }
      // Or initialize
      else {
        if (config["ssdim"].get_int() <= 0)
          throw std::string("The subspace dimensionality must be above zero!");
        
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
      es->set_hcl_optimization(&ls, &bfgs, config["hcl_line_cfg"].get_str(), config["hcl_bfgs_cfg"].get_str());
      new_pool.set_exponential_subspace(1, es);
    }
    

    // Load possible precomputed Gaussians

    // Set up bookkeeping for precomputed Gaussians
    std::vector<bool> already_computed;
    already_computed.resize(pool.size());
    for (unsigned int i=0; i<already_computed.size(); i++)
      already_computed[i] = false;

    // Go through the files
    if (config["coeffs"].specified) {
      std::ifstream coeffs_files(config["coeffs"].get_str().c_str());
      std::string coeff_file_name;
      while (coeffs_files >> coeff_file_name) {
        std::ifstream coeff_file(coeff_file_name.c_str());
        int g;
        while (coeff_file >> g) {

          already_computed[g]=true;

          if (config["to-pcgmm"].specified) {
            PrecisionConstrainedGaussian *pc = new PrecisionConstrainedGaussian(ps);
            pc->read(coeff_file);
            new_pool.set_pdf(g, pc);
          }

          else if (config["to-scgmm"].specified) {
            SubspaceConstrainedGaussian *sc = new SubspaceConstrainedGaussian(es);
            sc->read(coeff_file);
            new_pool.set_pdf(g, sc);
          }
        }
      }
    }

    // Go through every Gaussian in the pool
    Matrix covariance;
    Vector mean;
    for (int i=0; i<pool.size(); i++) {

      if (already_computed[i])
        continue;
      
      if (config["info"].get_int() > 0)
        std::cout << "Converting Gaussian: " << i << "/" << pool.size();
      
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
      
      // Get the old Gaussian parameters
      gaussian->get_covariance(covariance);
      gaussian->get_mean(mean);

      if (config["minvar"].specified)
      {
        double minvar = config["minvar"].get_float();
        for (int j = 0; j < gaussian->dim(); j++)
          if (covariance(j,j) < minvar)
            covariance(j,j) = minvar;
      }
        
      // Set parameters
      new_gaussian->set_parameters(mean, covariance);
      
      // Insert the converted Gaussian into the pool
      new_pool.set_pdf(i, new_gaussian);

      // Print kullback-leibler
      if (config["info"].get_int() > 0) {
        if (config["to-pcgmm"].specified || config["to-scgmm"].specified)
          std::cout << "\tkl-divergence: " << gaussian->kullback_leibler(*new_gaussian);
        std::cout << std::endl;
      }      
    }

    // Write out
    new_pool.write_gk(config["out"].get_str());
  }
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  } 
}
