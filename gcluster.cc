#include <iostream>
#include <fstream>
#include "Distributions.hh"
#include "conf.hh"
#include "LinearAlgebra.hh"

conf::Config config;

bool diagonal = true;
int num_clusters;
int num_iterations;
int dim;
int info;

struct GaussianInfo {
  Gaussian *g; // Used only if diagonal Gaussians are not in use

  Vector mean;
  Vector cov; // Used only if diagonal in use
  double ldet; // Logarithmic determinant (used only if diagonal in use)
  
  bool valid;

  GaussianInfo() : g(NULL), valid(false) { }
};

std::vector<GaussianInfo> gaussians;
std::vector<GaussianInfo> clusters;
std::vector<int> cluster_map;


void fill_random_permutation(int num, std::vector<int> &p)
{
  p.resize(num);
  for (int i = 0; i < num; i++)
    p[i] = i;
  for (int i = 0; i < num; i++)
  {
    int pos = i+(rand()%(num-i));
    int temp = p[i];
    p[i] = p[pos];
    p[pos] = temp;
  }
}


void compute_cluster_statistics(void)
{
  if (diagonal)
  {
    std::vector<int> gauss_count;

    gauss_count.resize(num_clusters);
    
    // Initialize statistics
    for (int i = 0; i < num_clusters; i++)
    {
      gauss_count[i] = 0;
      clusters[i].mean.resize(dim, 1);
      clusters[i].cov.resize(dim, 1);
      clusters[i].mean = 0;
      clusters[i].cov = 0;
    }
    // Accumulate Gaussians
    for (int i = 0; i < (int)gaussians.size(); i++)
    {
      Blas_Add_Mult(clusters[cluster_map[i]].mean, 1, gaussians[i].mean);
      Blas_Add_Mult(clusters[cluster_map[i]].cov, 1, gaussians[i].cov);
      gauss_count[cluster_map[i]]++;
    }
    // Normalize
    for (int i = 0; i < num_clusters; i++)
    {
      if (gauss_count[i] > 0)
      {
        double scale = 1/(double)gauss_count[i];
        Blas_Scale(scale, clusters[i].mean);
        Blas_Scale(scale, clusters[i].cov);
        clusters[i].valid = true;
        double t = 0;
        for (int j = 0; j < dim; j++)
          t += log(clusters[i].cov(j));
        clusters[i].ldet = t;
      }
      else
        clusters[i].valid = false;
    }
  }
  else
  {
    std::vector< std::vector<double> > weights;
    std::vector< std::vector<const Gaussian*> > temp_gauss;
    weights.resize(num_clusters);
    temp_gauss.resize(num_clusters);

    for (int i = 0; i < (int)gaussians.size(); i++)
    {
      weights[cluster_map[i]].push_back(1);
      temp_gauss[cluster_map[i]].push_back(gaussians[i].g);
    }

    for (int i = 0; i < num_clusters; i++)
    {
      if (weights[i].size() > 0)
      {
        if (clusters[i].g == NULL)
          clusters[i].g = new FullCovarianceGaussian(dim);
        else
          clusters[i].g->reset(dim);
        clusters[i].g->merge(weights[i], temp_gauss[i], false);
        clusters[i].valid = true;
      }
      else
      {
        if (clusters[i].g != NULL)
        {
          delete clusters[i].g;
          clusters[i].g = NULL;
        }
        clusters[i].valid = false;
      }
    }
  }
}


void make_initial_clusters(void)
{
  cluster_map.resize((int)gaussians.size());
  // Start with random Gaussians as centers
  std::vector<int> perm;
  fill_random_permutation((int)gaussians.size(), perm);
  for (int i = 0; i < num_clusters; i++)
    clusters[i].mean = gaussians[perm[i]].mean;

  // Cluster the Gaussians according to Euclidian distance
  for (int i = 0; i < (int)gaussians.size(); i++)
  {
    double min_dist = 1e100;
    int min_index = 0;
    for (int j = 0; j < num_clusters; j++)
    {
      Vector diff_mean = gaussians[i].mean;
      Blas_Add_Mult(diff_mean, -1, clusters[j].mean);
      double new_dist = Blas_Norm2(diff_mean);
      if (new_dist < min_dist)
      {
        min_dist = new_dist;
        min_index = j;
      }
    }
    cluster_map[i] = min_index;
  }
  compute_cluster_statistics();
}


double kl_divergence(int gauss_index, int cluster_index)
{
  if (diagonal)
  {
    double dist = 0;
    for (int i = 0; i < dim; i++)
    {
      double temp = gaussians[gauss_index].mean(i) -
        clusters[cluster_index].mean(i);
      dist += (gaussians[gauss_index].cov(i)+temp*temp) /
        clusters[cluster_index].cov(i);
    }
    return (clusters[cluster_index].ldet - gaussians[gauss_index].ldet +
            dist - dim)/2.0;
  }
  return gaussians[gauss_index].g->kullback_leibler(*clusters[cluster_index].g);
}


void save_clustering(const std::string &filename)
{
  // Determine the number of valid clusters and make the index map
  std::vector<int> index_map;
  int num_valid_clusters = 0;

  for (int i = 0; i < num_clusters; i++)
  {
    if (clusters[i].valid)
    {
      index_map.push_back(num_valid_clusters);
      num_valid_clusters++;
    }
    else
      index_map.push_back(-1);
  }

  if (num_valid_clusters == 0)
    throw std::string("No valid clusters!");
  
  std::ofstream out(filename.c_str());
  if (!out)
    throw std::string("Could not open file ") + filename;

  out << num_valid_clusters << "\n";
  for (int i = 0; i < (int)gaussians.size(); i++)
    out << i << " " << index_map[cluster_map[i]] << "\n";

  if (info > 0)
    printf("Wrote %i clusters\n", num_valid_clusters);
  
  if (!out)
    throw std::string("Error writing file: ") + filename;

}


int
main(int argc, char *argv[])
{
  try {
    config("usage: gcluster [OPTION...]\n")
      ('h', "help", "", "", "display help")
      ('g', "gk=FILE", "arg must", "", "gaussian definitions")
      ('o', "out=FILE", "arg must", "", "cluster file")
      ('F', "full", "", "", "use full statistics (much slower!)")
      ('C', "clusters=INT", "arg", "1000", "number of clusters (default 1000)")
      ('t', "iterations=INT", "arg", "4", "number of iterations (default 4)")
      ('i', "info=INT", "arg", "0", "info level")
      ;
    config.default_parse(argc, argv);

    info = config["info"].get_int();

    PDFPool *pool = new PDFPool;
    pool->read_gk(config["gk"].get_str());

    dim = pool->dim();

    num_clusters = config["clusters"].get_int();
    if (num_clusters < 2)
      throw std::string("Invalid number of clusters");
    num_iterations = config["iterations"].get_int();
    if (num_iterations < 1)
      throw std::string("Invalid number of iterations");

    gaussians.resize(pool->size());
    clusters.resize(num_clusters);

    // Initialize the Gaussian info
    if (config["full"].specified)
      diagonal = false;
    int num_valid_gaussians = 0;
    for (int i=0; i<pool->size(); i++)
    {
      Gaussian *g = dynamic_cast< Gaussian* > (pool->get_pdf(i));
      if (g == NULL)
      {
        fprintf(stderr, "Warning: Clustering operates only on Gaussian distributions!\n");
        gaussians[i].valid = false;
        continue;
      }
      if (diagonal)
      {
        g->get_covariance(gaussians[i].cov);
        double t = 0;
        for (int j = 0; j < dim; j++)
          t += log(gaussians[i].cov(j));
        gaussians[i].ldet = t;
      }
      else
      {
        gaussians[i].g = g;
      }
      g->get_mean(gaussians[i].mean);
      gaussians[i].valid = true;
      num_valid_gaussians++;
    }

    if (diagonal)
    {
      // We don't need the pool anymore
      delete pool;
      pool = NULL;
    }

    if (num_valid_gaussians < num_clusters)
      throw std::string("Not enough Gaussians to cluster!");
    make_initial_clusters();

    for (int iter = 0; iter < num_iterations; iter++)
    {
      double total_kl = 0;
      for (int i = 0; i < (int)gaussians.size(); i++)
      {
        double min_dist = 1e100;
        int min_index = 0;
        for (int j = 0; j < num_clusters; j++)
        {
          if (clusters[j].valid)
          {
            double new_dist = kl_divergence(i, j);
            if (new_dist < min_dist)
            {
              min_dist = new_dist;
              min_index = j;
            }
          }
        }
        if (info > 1)
          printf("Gaussian %i in cluster %i, distance %g\n", i, min_index,
                 min_dist);
        cluster_map[i] = min_index;
        total_kl += min_dist;
      }
      compute_cluster_statistics();

      if (info > 0)
        printf("Iteration %i: Average Kullback-Leibler divergence = %g\n",
               iter+1, total_kl/(double)gaussians.size());
    }
    
    if (!diagonal)
    {
      delete pool;
      pool = NULL;
    }

    save_clustering(config["out"].get_str());
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
