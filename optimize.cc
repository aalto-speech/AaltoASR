#include "Distributions.hh"
#include "conf.hh"
#include "LinearAlgebra.hh"
#include "HmmSet.hh"
#include "str.hh"


std::string stat_file;
conf::Config config;
HmmSet model;


int
main(int argc, char *argv[])
{
  try {
    config("usage: optimize [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('b', "base=BASENAME", "arg", "", "Previous base filename for model files")
      ('g', "gk=FILE", "arg", "", "Previous mixture base distributions")
      ('m', "mc=FILE", "arg", "", "Previous mixture coefficients for the states")
      ('p', "ph=FILE", "arg", "", "Previous HMM definitions")
      ('o', "out=FILE", "arg must", "", "base output file for the coefficients (will be appended with _bindex)")
      ('L', "list=LISTNAME", "arg", "", "file with one statistics file per line")
      ('\0', "subspace=FILE", "arg", "", "use an already initialized subspace")
      ('\0', "to-pcgmm", "", "", "convert Gaussians to have a subspace constraint on precisions")
      ('\0', "to-scgmm", "", "", "convert Gaussians to have an exponential subspace constraint")
      ('\0', "ml", "", "", "maximum likelihood estimation")
      ('\0', "mmi", "", "", "maximum mutual information estimation")
      ('\0', "minvar=FLOAT", "arg", "0.1", "minimum variance (default 0.1)")
      ('\0', "covsmooth", "arg", "0", "covariance smoothing (default 0.0)")
      ('\0', "C1=FLOAT", "arg", "1.0", "constant \"C1\" for MMI updates (default 1.0)")
      ('\0', "C2=FLOAT", "arg", "2.0", "constant \"C2\" for MMI updates (default 2.0)")
      ('\0', "ismooth=FLOAT", "arg", "0.0", "I-smoothing constant for discriminative training (default 0.0)")
      ('\0', "hcl_bfgs_cfg=FILE", "arg", "", "configuration file for HCL biconjugate gradient algorithm")
      ('\0', "hcl_line_cfg=FILE", "arg", "", "configuration file for HCL line search algorithm")
      ('B', "batch=INT", "arg", "0", "number of batch processes with the same recipe")
      ('I', "bindex=INT", "arg", "0", "batch process index")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    // Load the previous models
    if (config["base"].specified)
      model.read_all(config["base"].get_str());
    else {
      if (config["gk"].specified)
        model.read_gk(config["gk"].get_str());
      else
        throw std::string("At least --gk should be defined");
      if (config["mc"].specified)
        model.read_mc(config["mc"].get_str());
      if (config["ph"].specified)
        model.read_ph(config["ph"].get_str());        
    }
    
    // Linesearch for subspace models
    HCL_LineSearch_MT_d ls;
    if (config["hcl_line_cfg"].specified)
      ls.Parameters().Merge(config["hcl_line_cfg"].get_str().c_str());
    
    // lmBFGS for subspace models
    HCL_UMin_lbfgs_d bfgs(&ls);
    if (config["hcl_bfgs_cfg"].specified)
      bfgs.Parameters().Merge(config["hcl_bfgs_cfg"].get_str().c_str());
    
    model.set_hcl_optimization(&ls, &bfgs, config["hcl_line_cfg"].get_str(), config["hcl_bfgs_cfg"].get_str());
    
    // Open the file for writing out the subspace coefficients
    std::string outfilename = config["out"].get_str() + "_" + str::fmt(256, "%d", config["bindex"].get_int());
    std::ofstream outfile(outfilename.c_str());
    if (!outfile)
    {
      fprintf(stderr, "Could not open %s for writing\n", config["out"].get_str().c_str());
      exit(1);
    }
    
    // Optimize coefficients for this batch
    int start_pos = int(floor( (config["bindex"].get_int()-1) * model.num_pool_pdfs() / config["batch"].get_int() ));
    int end_pos = int(ceil( config["bindex"].get_int() * model.num_pool_pdfs() / config["batch"].get_int() ));

    // Print out some information
    if (config["info"].get_int()>0)
      std::cout << "Processing Gaussians " << start_pos+1 << "-" << end_pos
                << " of " << model.num_pool_pdfs() << std::endl;
    
    // Convert based on the statistics
    if (config["list"].specified) {

      if (!config["base"].specified &&
          !(config["gk"].specified && config["mc"].specified && config["ph"].specified))
        throw std::string("Must give either --base or all --gk, --mc and --ph");
      
      if (config["mmi"].specified && config["ml"].specified)
        throw std::string("Don't define both --ml and --mmi!");
      
      if (!config["mmi"].specified && !config["ml"].specified)
        throw std::string("Define either --ml or --mmi!");

      // ML or MMI?
      if (config["ml"].specified)
        model.set_estimation_mode(PDF::ML);
      else
        model.set_estimation_mode(PDF::MMI);
      
      // Open the list of statistics files
      std::ifstream filelist(config["list"].get_str().c_str());
      if (!filelist)
      {
        fprintf(stderr, "Could not open %s\n", config["list"].get_str().c_str());
        exit(1);
      }
      
      // Set parameters for Gaussian estimation
      model.set_gaussian_parameters(config["minvar"].get_double(),
                                    config["covsmooth"].get_double(),
                                    config["C1"].get_double(),
                                    config["C2"].get_double(),
                                    config["ismooth"].get_double());

      // Accumulate .gk statistics
      model.start_accumulating();
      while (filelist >> stat_file && stat_file != " ")
        model.accumulate_gk_from_dump(stat_file+".gks");
      
      for (int g=start_pos; g<end_pos; g++)
      {
        PrecisionConstrainedGaussian *pc = dynamic_cast< PrecisionConstrainedGaussian* > (model.get_pool_pdf(g));
        SubspaceConstrainedGaussian *sc = dynamic_cast< SubspaceConstrainedGaussian* > (model.get_pool_pdf(g));
        
        if (pc != NULL || sc != NULL)
          std::cout << "Training Gaussian: " << g+1 << "/" << model.num_pool_pdfs() << std::endl;
        
        try {
          if (pc != NULL)
            pc->estimate_parameters(config["minvar"].get_double(),
                                    config["covsmooth"].get_double(),
                                    config["C1"].get_double(),
                                    config["C2"].get_double(),
                                    config["ismooth"].get_double());
          else if (sc != NULL)
            sc->estimate_parameters(config["minvar"].get_double(),
                                    config["covsmooth"].get_double(),
                                    config["C1"].get_double(),
                                    config["C2"].get_double(),
                                    config["ismooth"].get_double());
        } catch (std::string errstr) {
          std::cout << "Warning: Gaussian number " << g
                    << ": " << errstr << std::endl;
        }
        
        outfile << g;
        if (pc != NULL) {
          pc->write(outfile);
        }
        else if (sc != NULL) {
          sc->write(outfile);
        }
        outfile << std::endl;
      }
    }


    // Convert the old model
    else {
      if (!config["to-pcgmm"].specified && !config["to-scgmm"].specified)
        throw std::string("Define either --to-pcgmm or --to-scgmm if you want to convert an old model!");
      if (config["to-pcgmm"].specified && config["to-scgmm"].specified)
        throw std::string("Don't define both --to-pcgmm and --to-scgmm if you want to convert an old model!");
      if (!config["subspace"].specified)
        throw std::string("Please specify --subspace if you want to convert an old model");

      PrecisionSubspace *ps;
      ExponentialSubspace *es;
      
      if (config["to-pcgmm"].specified) {
        ps = new PrecisionSubspace();
        std::ifstream in(config["subspace"].get_str().c_str());
        ps->read_subspace(in);
        ps->set_hcl_optimization(&ls, &bfgs, config["hcl_line_cfg"].get_str(), config["hcl_bfgs_cfg"].get_str());
        in.close();
      }
      else if (config["to-scgmm"].specified) {
        es = new ExponentialSubspace();
        std::ifstream in(config["subspace"].get_str().c_str());
        es->read_subspace(in);
        es->set_hcl_optimization(&ls, &bfgs, config["hcl_line_cfg"].get_str(), config["hcl_bfgs_cfg"].get_str());
        in.close();
      }

      Matrix covariance;
      Vector mean;
      for (int g=start_pos; g<end_pos; g++) {

        // Print Gaussian index
        if (config["info"].get_int()>0)
          std::cout << "Converting Gaussian: " << g+1 << "/" << model.num_pool_pdfs();

        // Fetch source Gaussian
        Gaussian *source_gaussian = dynamic_cast< Gaussian* > (model.get_pool_pdf(g));
        if (source_gaussian == NULL)
          continue;

        Gaussian *target_gaussian;
        if (config["to-pcgmm"].specified)
          target_gaussian = new PrecisionConstrainedGaussian(ps);
        else if (config["to-scgmm"].specified)
          target_gaussian = new SubspaceConstrainedGaussian(es);
        
        // Get the old Gaussian parameters
        source_gaussian->get_covariance(covariance);
        source_gaussian->get_mean(mean);
        
        // Set parameters
        target_gaussian->set_parameters(mean, covariance);

        // Print kullback-leibler
        if (config["info"].get_int() > 0) {
          std::cout << "\tkl-divergence: " << source_gaussian->kullback_leibler(*target_gaussian);
          std::cout << std::endl;
        }   

        outfile << g;
        target_gaussian->write(outfile);
        outfile << std::endl;
      }

      outfile.close();
    }
  }  
  
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  }  
  catch (std::string &str) {
    fprintf(stderr, "exception: %s\n", str.c_str());
    abort();
  }
} 




  
