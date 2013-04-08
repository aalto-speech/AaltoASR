#include <fstream>
#include <string>
#include <iostream>
#include <math.h>
#include <algorithm>

#include "io.hh"
#include "str.hh"
#include "conf.hh"
#include "HmmSet.hh"
#include "util.hh"

using namespace aku;

conf::Config config;


HmmSet model1;
HmmSet model2;


int
main(int argc, char *argv[])
{
  std::string m1_gk, m1_mc, m1_ph;
  std::string m2_gk, m2_mc;
  std::string out_base;
  try {
    config("usage: clskld [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('\0', "base1=BASENAME", "arg", "", "base filename for the source model")
      ('\0', "gk1=FILE", "arg", "", "Mixture base distributions for the source model")
      ('\0', "mc1=FILE", "arg", "", "Mixture coefficients for the states of the source model")
      ('\0', "ph1=FILE", "arg", "", "HMM definitions for the source model")
      ('\0', "base2=BASENAME", "arg", "", "base filename for the updated model")
      ('\0', "gk2=FILE", "arg", "", "Mixture base distributions for the updated model")
      ('\0', "mc2=FILE", "arg", "", "Mixture coefficients for the states of the updated model")
      ('\0', "ph2=FILE", "arg", "", "HMM definitions for the updated model")
      ('w', "mixtures", "", "", "Print KLDs of mixture weights")
      ('g', "gaussians", "", "", "Print KLDs of Gaussians")
      ('m', "means", "", "", "Print KLDs of Gaussian means")
      ('c', "covs", "", "", "Print KLDs of Gaussian covariances")
      ('\0', "only-silence", "", "", "Print KLDs only from silence states")
      ('\0', "no-silence", "", "", "Do not print KLDs from silence states")
      ;
    config.default_parse(argc, argv);
    
    // Load the first model
    if (config["base1"].specified)
    {
      model1.read_all(config["base1"].get_str());
    }
    else if (config["gk1"].specified && config["mc1"].specified &&
             config["ph1"].specified)
    {
      model1.read_gk(config["gk1"].get_str());
      model1.read_mc(config["mc1"].get_str());
      model1.read_ph(config["ph1"].get_str());
    }
    else
      throw std::string("Must give either --base1 or all --gk1, --mc1 and --ph1");

    // Load the second model
    if (config["base2"].specified)
    {
      model2.read_all(config["base2"].get_str());
    }
    else if (config["gk2"].specified && config["mc2"].specified &&
             config["ph2"].specified)
    {
      model2.read_gk(config["gk2"].get_str());
      model2.read_mc(config["mc2"].get_str());
      model2.read_ph(config["ph2"].get_str());
    }
    else
      throw std::string("Must give either --base2 or all --gk2, --mc2 and --ph2");

    // Model similarity check
    if (model1.num_emission_pdfs() != model2.num_emission_pdfs())
      throw std::string("Both models must have the same number of mixtures");
    if (model1.num_pool_pdfs() != model2.num_pool_pdfs())
      throw std::string("Both models must have the same number of Gaussians");

    std::vector<bool> mixture_print_flag;
    std::vector<bool> gaussian_print_flag;

    if (config["only-silence"].specified || config["no-silence"].specified)
    {
      mixture_print_flag.resize(model1.num_emission_pdfs(), false);
      gaussian_print_flag.resize(model1.num_pool_pdfs(), false);
      for (int i = 0; i < model1.num_hmms(); i++)
      {
        Hmm &hmm = model1.hmm(i);
        bool print_flag = false;
        if (hmm.label[0] == '_' &&
            hmm.label.find('-') == std::string::npos &&
            hmm.label.find('+') == std::string::npos)
        {
          // Silence state
          if (config["only-silence"].specified)
            print_flag = true;
        }
        else if (config["no-silence"].specified)
          print_flag = true;

        if (print_flag)
        {
          for (int j = 0; j < hmm.num_states(); j++)
          {
            int pdf_index = model1.state(hmm.state(j)).emission_pdf;
            mixture_print_flag[pdf_index] = print_flag;
            Mixture *m = model1.get_emission_pdf(pdf_index);
            for (int k = 0; k < (int)m->size(); k++)
            {
              int gaussian_index = m->get_base_pdf_index(k);
              gaussian_print_flag[gaussian_index] = print_flag;
            }
          }
        }
      }
    }
    

    if (config["mixtures"].specified)
    {
      for (int i = 0; i < model1.num_emission_pdfs(); i++)
      {
        if (mixture_print_flag.size() > (unsigned int)i &&
            !mixture_print_flag[i])
          continue;
        Mixture *m1 = model1.get_emission_pdf(i);
        Mixture *m2 = model2.get_emission_pdf(i);
        //assert( m1->size() == m2->size() );
        if (m1->size() != m2->size())
          continue;

        double kld = 0;
        for (int j = 0; j < m1->size(); j++)
        {
          double w1 = m1->get_mixture_coefficient(j);
          double w2 = m2->get_mixture_coefficient(j);
          kld += w2*log(w2/w1);
        }
        printf("%g\n", kld);
      }
    }
    if (config["gaussians"].specified)
    {
      PDFPool *pool1 = model1.get_pool();
      PDFPool *pool2 = model2.get_pool();

      for (int i = 0; i < pool1->size(); i++)
      {
        if (gaussian_print_flag.size() > (unsigned int)i &&
            !gaussian_print_flag[i])
          continue;

        Gaussian *pdf1 = dynamic_cast< Gaussian* >(pool1->get_pdf(i));
        Gaussian *pdf2 = dynamic_cast< Gaussian* >(pool2->get_pdf(i));
        if (pdf1 == NULL || pdf2 == NULL)
          throw std::string("Only Gaussian PDFs are supported!");

        Vector mean1;
        Vector mean2;
        Vector cov1;
        Vector cov2;
        pdf1->get_mean(mean1);
        pdf2->get_mean(mean2);
        pdf1->get_covariance(cov1);
        pdf2->get_covariance(cov2);
        int dim = cov1.size();
        double kld = 0;
        for (int j = 0; j < dim; j++)
        {
          double d = mean2(j)-mean1(j);
          kld += d*d/cov1(j);
          kld += cov2(j)/cov1(j) + log(cov1(j)/cov2(j));
        }
        kld = (kld - dim)/2.0;
        printf("%g\n", kld);
      }
    }
    
    if (config["means"].specified)
    {
      PDFPool *pool1 = model1.get_pool();
      PDFPool *pool2 = model2.get_pool();

      for (int i = 0; i < pool1->size(); i++)
      {
        if (gaussian_print_flag.size() > (unsigned int)i &&
            !gaussian_print_flag[i])
          continue;

        Gaussian *pdf1 = dynamic_cast< Gaussian* >(pool1->get_pdf(i));
        Gaussian *pdf2 = dynamic_cast< Gaussian* >(pool2->get_pdf(i));
        if (pdf1 == NULL || pdf2 == NULL)
          throw std::string("Only Gaussian PDFs are supported!");

        Vector mean1;
        Vector mean2;
        Vector cov;
        pdf1->get_mean(mean1);
        pdf2->get_mean(mean2);
        pdf1->get_covariance(cov);

        int dim = mean1.size();
        double kld = 0;
        for (int j = 0; j < dim; j++)
        {
          double d = mean2(j)-mean1(j);
          kld += d*d/cov(j);
        }
        kld /= 2.0;
        printf("%g\n", kld);
      }
    }

    if (config["covs"].specified)
    {
      PDFPool *pool1 = model1.get_pool();
      PDFPool *pool2 = model2.get_pool();

      for (int i = 0; i < pool1->size(); i++)
      {
        if (gaussian_print_flag.size() > (unsigned int)i &&
            !gaussian_print_flag[i])
          continue;

        Gaussian *pdf1 = dynamic_cast< Gaussian* >(pool1->get_pdf(i));
        Gaussian *pdf2 = dynamic_cast< Gaussian* >(pool2->get_pdf(i));
        if (pdf1 == NULL || pdf2 == NULL)
          throw std::string("Only Gaussian PDFs are supported!");

        Vector cov1;
        Vector cov2;
        pdf1->get_covariance(cov1);
        pdf2->get_covariance(cov2);

        int dim = cov1.size();
        double kld = 0;
        for (int j = 0; j < dim; j++)
          kld += cov2(j)/cov1(j) + log(cov1(j)/cov2(j));
        kld = (kld - dim)/2.0;
        printf("%g\n", kld);
      }
    }
  }

  // Handle errors
  catch (std::exception &e) {
    fprintf(stderr, "exception: %s\n", e.what());
    abort();
  }

  catch (std::string &str) {
    fprintf(stderr, "exception: %s\n", str.c_str());
    abort();
  }

}
