#include <iostream>
#include <fstream>
#include "Distributions.hh"
#include "conf.hh"
#include "LinearAlgebra.hh"
#include "HmmSet.hh"
#include "RegClassTree.hh"

typedef std::pair< double, std::pair<int, std::pair<int, int> > > queue_item_type;
typedef std::pair<std::pair<int,int>, double> merge_option_type;

conf::Config config;

bool diagonal = true;
int num_iterations;
int dim;
int info;

int num_clusters;

struct GaussianInfo {
  Gaussian *g; // Used only if diagonal Gaussians are not in use

  Vector mean;
  Vector cov; // Used only if diagonal in use
  double ldet; // Logarithmic determinant (used only if diagonal in use)
  
  bool valid;

  GaussianInfo() : g(NULL), valid(false) { }
};

class GaussianClustering {
public:
  std::vector<int> m_gaussian_ids;
std::vector<GaussianInfo> gaussians;
std::vector<GaussianInfo> clusters;
std::vector<int> cluster_map;

  std::vector<int> real_cluster_ids;
  int num_clusters;



  GaussianClustering(std::vector<int> &gaussian_ids) : m_gaussian_ids(gaussian_ids) { }

  void set_num_clusters(int n_clusters) { num_clusters = n_clusters; clusters.resize(n_clusters); }
  void make_initial_clusters(void);
  void collect_gaussians(PDFPool *pool);

  merge_option_type get_best_merge_option() const;

  void merge(std::pair<int,int> p);

  void refine_clustering(int num_iterations);

private:
  void compute_cluster_statistics(void);
  double kl_divergence(int gauss_index, int cluster_index) const;
  double kl_divergence(const GaussianInfo &g1, const GaussianInfo &g2) const;
};

std::vector<GaussianClustering> cluster_groups;


struct CompPair {
        bool operator()(queue_item_type p1, queue_item_type p2) {
          std::cerr << "hey" << std::endl;
          return p1.first > p2.first;
        }
    };


merge_option_type GaussianClustering::get_best_merge_option() const {
  std::pair<int,int> best_merge_option (0,0);
  double smallest_distance = 1e100;

  for(unsigned int i = 0; i < clusters.size(); ++i) {
    if(!clusters[i].valid) continue;
    for(unsigned int j = i+1; j < clusters.size(); ++j) {
      if(!clusters[j].valid) continue;

      double dist = kl_divergence(clusters[i], clusters[j]);
      if(dist < smallest_distance) {
        smallest_distance = dist;
        best_merge_option.first = i;
        best_merge_option.second = j;
      }
    }
  }

  merge_option_type merge;
  merge.first = best_merge_option;
  merge.second = smallest_distance;
  return merge;
}

void GaussianClustering::merge(std::pair<int,int> p) {
  clusters[p.second].valid = false;
  for(unsigned int g = 0; g < cluster_map.size(); ++g)
    if(cluster_map[g] == p.second) cluster_map[g] = p.first;

  compute_cluster_statistics();

}

void GaussianClustering::collect_gaussians(PDFPool *pool)
{
  gaussians.resize(m_gaussian_ids.size());
  for (unsigned int i = 0; i < m_gaussian_ids.size(); i++) {
    Gaussian *g = dynamic_cast<Gaussian*> (pool->get_pdf(m_gaussian_ids[i]));
    if (g == NULL) {
      fprintf(stderr,
          "Warning: Clustering operates only on Gaussian distributions!\n");
      gaussians[i].valid = false;
      continue;
    }
    if (diagonal) {
      g->get_covariance(gaussians[i].cov);
      double t = 0;
      for (int j = 0; j < dim; j++)
        t += log(gaussians[i].cov(j));
      gaussians[i].ldet = t;
    }
    else {
      gaussians[i].g = g;
    }
    g->get_mean(gaussians[i].mean);
    gaussians[i].valid = true;
  }
}

void GaussianClustering::refine_clustering(int num_iterations) {
  for (int iter = 0; iter < num_iterations; iter++) {
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
}

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


void GaussianClustering::compute_cluster_statistics(void)
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


void GaussianClustering::make_initial_clusters(void)
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


double
GaussianClustering::kl_divergence(int gauss_index, int cluster_index) const
{
  return kl_divergence(gaussians[gauss_index], clusters[cluster_index]);
}

double
GaussianClustering::kl_divergence(const GaussianInfo &g1, const GaussianInfo &g2) const {
  if (diagonal)
  {
    double dist = 0;
    for (int i = 0; i < dim; i++)
    {
      double temp = g1.mean(i) - g2.mean(i);
      dist += (g1.cov(i)+temp*temp) / g2.cov(i);
    }
    return (g2.ldet - g1.ldet + dist - dim)/2.0;
  }
  return g1.g->kullback_leibler(*g2.g);
}


void save_clustering(const std::string &filename)
{
  std::map<int, int> gauss_to_cluster;

  int next_cluster_id = 0;

  for(unsigned int i = 0; i < cluster_groups.size(); ++i) {
    GaussianClustering &gc = cluster_groups[i];
    gc.real_cluster_ids.clear();
    for(unsigned int j = 0; j < gc.clusters.size(); ++j) {
      if(gc.clusters[j].valid) gc.real_cluster_ids.push_back(next_cluster_id++);
      else gc.real_cluster_ids.push_back(-1);
  }

    for(unsigned int g = 0; g < gc.gaussians.size(); ++g) {
      gauss_to_cluster[gc.m_gaussian_ids[g]] = gc.real_cluster_ids[gc.cluster_map[g]];
    }
  }

    if (next_cluster_id == 0)
    throw std::string("No valid clusters!");
  
  std::ofstream out(filename.c_str());
  if (!out)
    throw std::string("Could not open file ") + filename;

  out << next_cluster_id << "\n";

  for(unsigned int g= 0; g < gauss_to_cluster.size(); ++g)
    out << g << " " << gauss_to_cluster[g] << "\n";

  if (info > 0)
    printf("Wrote %i clusters\n", next_cluster_id);
  
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
      ('R', "regtree=FILE", "arg", "", "regression tree file, if given, the clustering will group gaussians from the same treenode together")
      ('b', "base=BASENAME", "arg", "", "base filename for model files, only necessary if regtree is given")
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

    RegClassTree *rtree = NULL;
    HmmSet *model = NULL;

    if(config["regtree"].specified && config["base"].specified ) { // Read tree
      model = new HmmSet();
      model->read_all(config["base"].get_str());
      std::ifstream in(config["regtree"].get_c_str());
      rtree = new RegClassTree();
      rtree->read(&in, model);

      std::vector<RegClassTree::Node*> node_v;
      rtree->get_terminal_nodes(node_v);

      for(unsigned int i = 0; i < node_v.size(); ++i) {
        std::set<int> indices;
        node_v[i]->get_pdf_indices(model, indices);
        std::vector<int> indices_v;
        for(std::set<int>::iterator it = indices.begin(); it != indices.end(); ++it)
          indices_v.push_back(*it);
        cluster_groups.push_back(GaussianClustering(indices_v));
      }
      delete model;

    } else {
      if (config["regtree"].specified || config["base"].specified) throw std::string("Both tree and model must be given");

      std::vector<int> indices_v;
      indices_v.resize(pool->size());
      for(int i = 0; i < pool->size(); ++i) indices_v[i] = i;

      cluster_groups.push_back(GaussianClustering(indices_v));
    }

    if(cluster_groups.size() > 1) {
      int cluster_count = num_clusters * 2;
      int c = cluster_count / cluster_groups.size();
      for(unsigned int i = 1; i < cluster_groups.size(); ++i) {
        cluster_groups[i].set_num_clusters(std::min(c,(int)cluster_groups[i].m_gaussian_ids.size()));
        cluster_count -= std::min(c,(int)cluster_groups[i].m_gaussian_ids.size());
      }
      cluster_groups[0].set_num_clusters(std::min(cluster_count,(int)cluster_groups[0].m_gaussian_ids.size()));
    } else {
      if ((int)cluster_groups[0].m_gaussian_ids.size() < num_clusters)
        throw std::string("Not enough Gaussians to cluster!");
      cluster_groups[0].set_num_clusters(num_clusters);
    }


    // Initialize the Gaussian info
    if (config["full"].specified)
      diagonal = false;
    std::cerr << "make initial clusters" << std::endl;

    for(unsigned int i = 0; i < cluster_groups.size(); ++i) {
      cluster_groups[i].collect_gaussians(pool);
      cluster_groups[i].make_initial_clusters();
    }

    if (diagonal)
    {
      // We don't need the pool anymore
      delete pool;
      pool = NULL;
    }

    int num_total_clusters = 0;

    std::cerr << "start clustering" << std::endl;
    for(unsigned int i = 0; i < cluster_groups.size(); ++i) {
      cluster_groups[i].refine_clustering(4);
      num_total_clusters += cluster_groups[i].clusters.size();
    }

    if(cluster_groups.size() > 1) {
      int num_merges = 0;
      std::vector<int> group_num_merges;
      group_num_merges.resize(cluster_groups.size(), 0);

      std::priority_queue< queue_item_type ,std::vector< queue_item_type > , CompPair> queue;

      for(unsigned int i = 0; i < cluster_groups.size(); ++i) {
        int c = 0;
        for(unsigned int j = 0; j < cluster_groups[i].clusters.size(); ++j) {
          if(cluster_groups[i].clusters[j].valid) ++c;
        }
      }

      for(unsigned int i = 0; i < cluster_groups.size(); ++i) {
        merge_option_type merge_option = cluster_groups[i].get_best_merge_option();
        queue_item_type queue_item;
        queue_item.first = merge_option.second;
        queue_item.second.first = i;
        queue_item.second.second = merge_option.first;
        queue.push(queue_item);
      }

      while(num_total_clusters > num_clusters) {
        queue_item_type queue_item;
        queue_item = queue.top();
        queue.pop();

        cluster_groups[queue_item.second.first].merge(queue_item.second.second);
        std::pair<std::pair<int,int>, double> merge_option = cluster_groups[queue_item.second.first].get_best_merge_option();
        std::cerr << "Merge in group" << queue_item.second.first << std::endl;
        queue_item.first = merge_option.second;
        queue_item.second.second = merge_option.first;
        queue.push(queue_item);

        ++num_merges;
        ++group_num_merges[queue_item.second.first];
        --num_total_clusters;

        if(group_num_merges[queue_item.second.first] > (num_clusters / num_iterations / (int)group_num_merges.size())) {
          group_num_merges[queue_item.second.first] = 0;
          cluster_groups[queue_item.second.first].refine_clustering(2);
        }
      }

      for(unsigned int i = 0; i < cluster_groups.size(); ++i)
        if(group_num_merges[i] > 0) cluster_groups[i].refine_clustering(2);

      for(unsigned int i = 0; i < cluster_groups.size(); ++i) {
        int c = 0;
        for(unsigned int j = 0; j < cluster_groups[i].clusters.size(); ++j) {
          if(cluster_groups[i].clusters[j].valid) ++c;
        }
      }

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
