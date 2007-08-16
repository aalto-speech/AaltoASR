#include "Distributions.hh"
#include "conf.hh"
#include "LinearAlgebra.hh"

conf::Config config;
PrecisionSubspace *ps;
ExponentialSubspace *es;


void bfgs_set_defaults(HCL_UMin_lbfgs_d &bfgs) {
  bfgs.Parameters().PutValue("MaxItn", 100);
  bfgs.Parameters().PutValue("Typf", 1.0);
  bfgs.Parameters().PutValue("TypxNorm", 1.0);
  bfgs.Parameters().PutValue("GradTol", 1.0e-2);
  bfgs.Parameters().PutValue("MinStep", 1e-20);
  bfgs.Parameters().PutValue("MaxStep", 1e+20);
  bfgs.Parameters().PutValue("CscMaxLimit", 5);
  bfgs.Parameters().PutValue("MaxUpdates", 4);
}


void line_set_defaults(HCL_LineSearch_MT_d &ls) {
  ls.Parameters().PutValue("FcnDecreaseTol", 1e-4);
  ls.Parameters().PutValue("SlopeDecreaseTol", 9e-1);
  ls.Parameters().PutValue("MinStep", 1e-20);
  ls.Parameters().PutValue("MaxStep", 1e+20);
  ls.Parameters().PutValue("MaxSample", 8);
  ls.Parameters().PutValue("BracketIncrease", 4);
}


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
        std::string skip;
        in >> skip;
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
      ps->set_bfgs(&bfgs);
      new_pool.set_precision_subspace(1, ps);
    }
    
    // Initialize the Scgmm case
    if (config["to-scgmm"].specified) {

      es = new ExponentialSubspace(config["ssdim"].get_int(), pool.dim());

      // Read from file
      if (config["subspace"].specified) {
        std::ifstream in(config["subspace"].get_str().c_str());
        std::string skip;
        in >> skip;
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
      es->set_bfgs(&bfgs);
      new_pool.set_exponential_subspace(1, es);
    }
    
    // Go through every Gaussian in the pool
    Matrix covariance;
    Vector mean;
    for (int i=0; i<pool.size(); i++) {

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
      try {
        new_gaussian->set_parameters(mean, covariance);
      } catch(LaException &e) {
        // Try optimizing subspace parameters in 'safe mode' if things go bad
        bfgs_set_defaults(bfgs);
        line_set_defaults(ls);
        try {
        new_gaussian->set_parameters(mean, covariance);
        } catch(LaException e) {}	  
        if (config["hcl_line_cfg"].specified)
          ls.Parameters().Merge(config["hcl_line_cfg"].get_str().c_str());  
        if (config["hcl_bfgs_cfg"].specified)
          bfgs.Parameters().Merge(config["hcl_bfgs_cfg"].get_str().c_str());
      }

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
