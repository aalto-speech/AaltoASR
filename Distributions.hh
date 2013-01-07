#ifndef DISTRIBUTIONS_HH
#define DISTRIBUTIONS_HH

#include <queue>

#include "FeatureBuffer.hh"
#include "FeatureModules.hh"
#include "LinearAlgebra.hh"
#ifdef USE_SUBSPACE_COV
# include "Subspaces.hh"
#endif
#include "ziggurat.hh"
#include "mtw.hh"

// Bitmasks for statistics mode. Note! PDF_ML_FULL_STATS implies PDF_ML_STATS
#define PDF_ML_STATS      1
#define PDF_ML_FULL_STATS 2
#define PDF_MMI_STATS     4
#define PDF_MPE_NUM_STATS 8
#define PDF_MPE_DEN_STATS 16


namespace aku {

class PDF {
public:

  typedef int StatisticsMode; // Use bitflags above
  enum EstimationMode { ML_EST=1, MMI_EST=2, MPE_EST=3, MPE_MMI_PRIOR_EST=4 };
  enum AccumBuffer { ML_BUF=0, MMI_BUF=1, MPE_NUM_BUF=2, MPE_DEN_BUF=3 };
  
  // COMMON

  PDF() { m_update = true; }
  virtual ~PDF() {}
  /* The feature dimensionality */
  int dim() const { return m_dim; }
  
  // TRAINING

  /* Initializes the accumulator buffers */  
  virtual void start_accumulating(StatisticsMode mode) = 0;
  /** Has accumulating been started
   * \returns true if start_accumulating has been called, false otherwise
   */
  virtual bool is_accumulating() const = 0;
  /* Accumulates the statistics for this pdf */
  virtual void accumulate(double prior,
			  const Vector &f, 
			  int accum_pos = 0) = 0;
  virtual void accumulate_aux_gamma(double gamma, int accum_pos = 0) = 0;
  /* Writes the currently accumulated statistics to a file */
  virtual void dump_statistics(std::ostream &os) const = 0;
  /* Accumulates from a file dump */
  virtual void accumulate_from_dump(std::istream &is, StatisticsMode mode) = 0;
  /* Stops training and clears the accumulators */
  virtual void stop_accumulating() = 0;
  /* Tells if this pdf has been accumulated */
  virtual bool accumulated(int accum_pos = 0) const = 0;
  /* Use the accumulated statistics to update the current model parameters. */
  virtual void estimate_parameters(EstimationMode mode) = 0;
  /* Set the update mode */
  virtual void set_update_flag(bool update_flag) { m_update = update_flag; }
  
  // LIKELIHOODS
  
  /* The likelihood of the current feature given this model */
  virtual double compute_likelihood(const Vector &f) const = 0;
  /* The log likelihood of the current feature given this model */
  virtual double compute_log_likelihood(const Vector &f) const = 0;


  // SAMPLING

  /* Draw a random sample from this distribution */
  virtual void draw_sample(Vector &sample) = 0;
  
  // IO

  /* Write the parameters of this distribution to the stream os */
  virtual void write(std::ostream &os) const = 0;
  /* Read the parameters of this distribution from the stream is */
  virtual void read(std::istream &is) = 0;


protected:
  int m_dim;
  bool m_update; /// If false, refrain from re-estimating the parameters
};


class PDFPool {
public:
  
  PDFPool();
  PDFPool(int dim);
  virtual ~PDFPool();

  /** Set the dimensionality of the distributions in this pool
   * \param dim Feature/distribution dimension
   */
  void set_dim(int dim) { m_dim = dim; }
  /// The dimensionality of the distributions in this pool
  int dim() const { return m_dim; }
  /// Number of distributions in this pool
  int size() const { return m_pool.size(); };
  /// Reset everything
  void reset();
  
  /// Get the pdf from the position index
  PDF* get_pdf(int index) const;
  /// Set the pdf in the position index
  void set_pdf(int index, PDF *pdf);

  /** Add a new pdf to the pool.
   * \return Index of the pdf in the pool
   */
  int add_pdf(PDF *pdf);

  /** Deletes a pdf from the pool. Note that all indices after the deleted
   *  pdf will decrease by one.
   *  \param index PDF index
   */
  void delete_pdf(int index);
  
  /// Read the distributions from a .gk -file
  void read_gk(const std::string &filename);
  /// Write the distributions to a .gk -file
  void write_gk(const std::string &filename) const;
 
#ifdef BINARY_GAUSSIAN_FILES
  /// Read the distributions from a binary form .bgk -file
  void read_bgk(const std::string &filename);
  /// Write the distributions to a binary form .bgk -file
  void write_bgk(const std::string &filename) const;
#endif

  /// Reset the cache
  void reset_cache();

  /** Compute the likelihood of a feature for pdf in the pool. Uses cache.
   * \param f the feature vector
   * \param index the pdf index
   * \return the likelihood of the given feature for some pdf
   */
  double compute_likelihood(const Vector &f, int index);


  double compute_clustered_likelihood(const Vector &f, int index);

  /// \brief Computes likelihoods for all distributions to the cache.
  ///
  /// If Gaussian clustering is in use, computes likelihoods for all
  /// distributions in the best clusters, until either evaluate_min_clusters()
  /// or evaluate_min_gaussians() has been reached. For the rest of the
  /// clusters, uses the cluster center likelihood.
  ///
  /// \param f the feature vector
  ///
  void precompute_likelihoods(const Vector &f);

  /// Estimates parameters of the pdfs in the pool
  void estimate_parameters(PDF::EstimationMode mode);


  /********************************************************************/
  /* Gaussian specific methods                                        */
  /********************************************************************/
  
  /** Sets the parameters used in Gaussian estimation
   * \param minvar    Minimum diagonal variance term for Gaussians
   * \param covsmooth Covariance smoothing value
   * \param c1        EBW C1 constant
   * \param c2        EBW C2 constant
   * \param ismooth   I-smoothing constant
   * \param mmi_prior_ismooth I-smoothing for the prior MMI-model
   * \param ebw_max_kld Maximum KLD change in EBW update
   */
  void set_gaussian_parameters(double minvar = 0, double covsmooth = 0,
                               double c1 = 0, double c2 = 0,
                               double ismooth = 0, double mmi_prior_ismooth = 0,
                               double ebw_max_kld = 0);

  /// Sets I-smoothing prior mode
  void set_ismooth_prev_prior(bool prev) { m_ismooth_prev_prior = prev; }

#ifdef USE_SUBSPACE_COV
  /** Set the HCL objects and settings for optimization
   * \param ls            HCL linesearch
   * \param bfgs          HCL BFGS optimization algorithm
   * \param ls_cfg_file   HCL linesearch configuration file
   * \param bfgs_cfg_file HCL BFGS optimization algorithm configuration file
   */
  void set_hcl_optimization(HCL_LineSearch_MT_d *ls,
                            HCL_UMin_lbfgs_d *bfgs,
                            std::string ls_cfg_file,
                            std::string bfgs_cfg_file);
#endif
  
  /** Splits a Gaussian in the pool with some constrains
  * \param index     Index of the Gaussian to be split
  * \param new_index The index of the newly created Gaussian is saved
  *                  to this pointer
  * \return true if the split was succesful, false otherwise
  */
  bool split_gaussian(int index, int *new_index);

  double get_minvar(void) const { return m_minvar; }
  double get_covsmooth(void) const { return m_covsmooth; }

  /** Functions for accessing subspaces for Gaussian parameters */
#ifdef USE_SUBSPACE_COV
  void set_precision_subspace(int id, PrecisionSubspace *ps);
  void set_exponential_subspace(int id, ExponentialSubspace *es);
  PrecisionSubspace *get_precision_subspace(int id);
  ExponentialSubspace *get_exponential_subspace(int id);
  void remove_precision_subspace(int id);
  void remove_exponential_subspace(int id);
#endif 
  
  /** \param index PDF index
   * \return Gaussian occupancy, -1 if not a Gaussian or not accumulated. */
  double get_gaussian_occupancy(int index) const;

  void get_occ_sorted_gaussians(std::vector<int> &sorted_gaussians,
                                double minocc);

  struct Gaussian_occ_comp;
  
  /********************************************************************/
  /* Methods for Gaussian clustering                                  */
  /********************************************************************/

  bool use_clustering() { return m_use_clustering; }
  int number_of_clusters() { return m_number_of_clusters; }
  int evaluate_min_clusters() { return m_evaluate_min_clusters; }
  int evaluate_min_gaussians() { return m_evaluate_min_gaussians; }

  void set_use_clustering(bool use) { m_use_clustering = use; }
  void set_number_of_clusters(int n) { m_number_of_clusters = n; }

  /// \brief Specifies a minimum number of clusters to evaluate accurately
  /// before precompute_likelihoods() may use cluster center likelihoods.
  ///
  void set_evaluate_min_clusters(int n) { m_evaluate_min_clusters = n; }

  /// \brief Specifies a minimum number of Gaussians to evaluate accurately
  /// before precompute_likelihoods() may use cluster center likelihoods.
  ///
  void set_evaluate_min_gaussians(int n) { m_evaluate_min_gaussians = n; }

  /// \brief Reads clustering of the Gaussians from a file.
  ///
  void read_clustering(const std::string &filename);
  void write_cluster_gaussians(const std::string &filename);
  void inject_cluster_gaussians(PDFPool *target_pool);
  std::vector<PDF*> &get_cluster_centers() { return m_cluster_centers; }
  std::vector<std::vector<int> > &get_cluster_to_gaussians() { return m_cluster_to_gaussians; }
  
private:
  // Standard things
  std::vector<PDF*> m_pool;
  std::vector<double> m_likelihoods;
  std::vector<int> m_valid_likelihoods;
  int m_dim;

  // Estimation constants
  double m_minvar;
  double m_covsmooth;
  double m_c1;
  double m_c2;
  double m_ismooth;
  double m_mmi_prior_ismooth;
  bool m_ismooth_prev_prior;
  double m_ebw_max_kld;

#ifdef USE_SUBSPACE_COV
  // Subspaces
  std::map<int, PrecisionSubspace*> m_precision_subspaces;
  std::map<int, ExponentialSubspace*> m_exponential_subspaces;
#endif

  // Clustering
  bool m_use_clustering;
  std::vector<PDF*> m_cluster_centers;
  std::vector<int> m_gaussian_to_cluster;
  std::vector<std::vector<int> > m_cluster_to_gaussians;
  int m_number_of_clusters;
  int m_evaluate_min_clusters;
  int m_evaluate_min_gaussians;

  typedef std::pair<int,double> ClusterLikelihoodPair;
  struct cl_compare
  {
    bool operator()(ClusterLikelihoodPair cl1, ClusterLikelihoodPair cl2) const
    {
      return cl1.second < cl2.second;
    }
  };
  typedef std::priority_queue<ClusterLikelihoodPair, std::vector<ClusterLikelihoodPair>, cl_compare> ClusterLikelihoods;
};



class GaussianAccumulator {
public:
  virtual ~GaussianAccumulator() {}
  int dim() const { return m_dim; }
  bool accumulated() const { return m_accumulated; }
  int feacount() const { return m_feacount; }
  double gamma() const { return m_gamma; }
  double aux_gamma() const { return m_aux_gamma; }
  void set_gamma(double g) { m_gamma = g; }
  void set_aux_gamma(double g) { m_aux_gamma = g; }
  void get_mean_estimate(Vector &mean_estimate) const;
  void get_accumulated_mean(Vector &mean) const;
  void set_accumulated_mean(Vector &mean);
  virtual void get_covariance_estimate(Matrix &covariance_estimate) const = 0;
  virtual void get_accumulated_second_moment(Matrix &second_moment) const = 0;
  virtual void get_accumulated_second_moment(Vector &second_moment) const = 0;
  virtual void set_accumulated_second_moment(Matrix &second_moment) = 0;
  virtual void accumulate(int feacount, double gamma, const Vector &f) = 0;
  virtual void accumulate_aux_gamma(double gamma) { m_aux_gamma += gamma; }
  virtual void dump_statistics(std::ostream &os) const = 0;
  virtual void accumulate_from_dump(std::istream &is) = 0;
  virtual bool full_stats_accumulated() const = 0;
  virtual void reset() = 0;
protected:
  int m_dim;
  bool m_accumulated;
  int m_feacount;
  double m_gamma;
  double m_aux_gamma;
  Vector m_mean;
};


class FullStatisticsAccumulator : public GaussianAccumulator {
public:
  FullStatisticsAccumulator(int dim) { 
    m_mean.resize(dim);
    m_second_moment.resize(dim,dim);
    m_dim=dim;
    reset();
  }
  virtual void get_covariance_estimate(Matrix &covariance_estimate) const;
  virtual void get_accumulated_second_moment(Matrix &second_moment) const;
  virtual void get_accumulated_second_moment(Vector &second_moment) const;
  virtual void set_accumulated_second_moment(Matrix &second_moment);
  virtual void accumulate(int feacount, double gamma, const Vector &f);
  virtual void dump_statistics(std::ostream &os) const;
  virtual void accumulate_from_dump(std::istream &is);
  virtual bool full_stats_accumulated() const { return accumulated(); }
  virtual void reset();
private:
  SymmetricMatrix m_second_moment;
};


class DiagonalStatisticsAccumulator : public GaussianAccumulator {
public:
  DiagonalStatisticsAccumulator(int dim) {
    m_mean.resize(dim);
    m_second_moment.resize(dim);
    m_dim=dim;
    reset();
  }
  virtual void get_covariance_estimate(Matrix &covariance_estimate) const;
  virtual void get_accumulated_second_moment(Matrix &second_moment) const;
  virtual void get_accumulated_second_moment(Vector &second_moment) const;
  virtual void set_accumulated_second_moment(Matrix &second_moment);
  virtual void accumulate(int feacount, double gamma, const Vector &f);
  virtual void dump_statistics(std::ostream &os) const;
  virtual void accumulate_from_dump(std::istream &is);
  virtual bool full_stats_accumulated() const { return false; }
  virtual void reset();
private:
  Vector m_second_moment;
};




class Gaussian : public PDF {
public:
  
  class ConstrainedEBWSolver : public util::FuncEval {
  public:
    ConstrainedEBWSolver(const Gaussian &g, double m0_stat,
                         const Vector &m1_stat, const Matrix &m2_stat) :
      m_g(g), m_m0(m0_stat), m_m1(m1_stat), m_m2(m2_stat) { g.get_mean(m_mean0); g.get_covariance(m_cov0); }
    virtual double evaluate_function(double p); //!< Given D
    void solve_mean_and_cov(double d);
    double mean_kld(void); //!< Compares m_mean0 to m_new_mean
    double cov_kld(void); //!< Compares m_cov0 to m_new_cov
    
    double constrained_update(double min_d, double max_kld);

    void get_parameters(Vector &new_mean, Matrix &new_cov) const;

  protected:
    const Gaussian &m_g;
    const double m_m0;
    const Vector &m_m1;
    const Matrix &m_m2;
    
    Vector m_mean0;
    Matrix m_cov0;

    Vector m_new_mean;
    Matrix m_new_cov;
  };
  
public:

  Gaussian() { chol = NULL; m_ebw_max_kld = 0; m_fixed_d = -1; m_min_d = -1; m_realized_d = -1; };
  virtual ~Gaussian() { if (chol != NULL) delete chol; }
  // ABSTRACT FUNCTIONS, SHOULD BE OVERWRITTEN IN THE GAUSSIAN IMPLEMENTATIONS
  
  /* Resets the Gaussian to have dimensionality dim and all values zeroed */
  virtual void reset(int dim) = 0;
  
  // FROM PDF

  /* Initializes the accumulator buffers */  
  virtual void start_accumulating(StatisticsMode mode) = 0;
  virtual bool is_accumulating() const { return (m_accums.size()>0?true:false); }
  /* Accumulates the maximum likelihood statistics for the Gaussian
     weighed with a prior. */
  virtual void accumulate(double prior, const Vector &f, 
			  int accum_pos = 0) ;
  virtual void accumulate_aux_gamma(double gamma, int accum_pos = 0);
  /* Writes the currently accumulated statistics to a stream */
  virtual void dump_statistics(std::ostream &os) const;
  /* Accumulates from dump */
  virtual void accumulate_from_dump(std::istream &is, StatisticsMode mode);
  /* Stops training and clears the accumulators */
  virtual void stop_accumulating();
  /* Tells if this Gaussian has been accumulated */
  virtual bool accumulated(int accum_pos = 0) const;
  /* Use the accumulated statistics to update the current model parameters. */
  virtual void estimate_parameters(EstimationMode mode) { estimate_parameters(mode, 0, 0, 1, 2, 0, false); }
  virtual void estimate_parameters(EstimationMode mode, double minvar,
                                   double covsmooth, double c1, double c2,
                                   double tau, bool ml_stats_target);
  // GAUSSIAN SPECIFIC

  /// EBW update of the mean parameter
  void mean_ebw_update(const Vector &old_mean, double m0_stat,
                       const Vector &m1_stat, double d, Vector &new_mean) const;
  /// EBW update of the covariance parater
  void cov_ebw_update(const Matrix &old_cov, const Vector &old_mean,
                      const Vector &new_mean, double m0_stat,
                      const Matrix &m2_stat, double d, Matrix &new_cov) const;
  /// Sets the maximum KLD limit for EBW update
  void set_ebw_max_kld(double max_kld) { m_ebw_max_kld = max_kld; }

  /// Sets the fixed D for EBW updates
  void set_ebw_fixed_d(double d) { m_fixed_d = d; m_realized_d = d; }

  /// Return realized EBW D value
  double get_realized_d(void) { return m_realized_d; }

  /// Return minimum EBW D value
  double get_minimum_d(void) { return m_min_d; }
  
  /// Returns the mean vector for this Gaussian
  virtual void get_mean(Vector &mean) const = 0;
  /// Returns the covariance matrix for this Gaussian
  virtual void get_covariance(Matrix &covariance) const = 0;
  /// Sets the parameters for this Gaussian
  virtual void set_parameters(const Vector &mean,
                              const Matrix &covariance)
  { set_mean(mean); set_covariance(covariance); }
  /// Sets the mean vector for this Gaussian
  virtual void set_mean(const Vector &mean) = 0;
  /// Sets the covariance matrix for this Gaussian
  virtual void set_covariance(const Matrix &covariance,
                              bool finish_statistics = true) = 0;
  /// Get the diagonal of the covariance matrix
  virtual void get_covariance(Vector &covariance) const;
  /// Set the diagonal of the covariance matrix and off-diagonal to zero
  virtual void set_covariance(const Vector &covariance,
                              bool finish_statistics = true);
  /// Returns a copy of this Gaussian object
  virtual Gaussian* copy_gaussian(void) = 0;
  /// The likelihood of the current feature given this model using exponential feature vector
  virtual double compute_likelihood_exponential(const Vector &exponential_feature) const = 0;
  /// The log likelihood of the current feature given this model using exponential feature vector
  virtual double compute_log_likelihood_exponential(const Vector &exponential_feature) const = 0;
  
  // THESE FUNCTIONS HAVE ALSO A COMMON IMPLEMENTATION, BUT CAN BE OVERWRITTEN

  /// Splits the current Gaussian to two by disturbing the mean
  virtual void split(Gaussian &s1, Gaussian &s2, double perturbation = 0.2) const;
  /// Splits the current Gaussian to two by disturbing the mean
  virtual void split(Gaussian &s2, double perturbation = 0.2);
  /// Sets the parameters for the current Gaussian by merging m1 and m2
  virtual void merge(double w1, const Gaussian &m1,
                     double w2, const Gaussian &m2,
                     bool finish_statistics = true);
  /** Sets the parameters for the current Gaussian by merging the Gaussians
      given in a vector with the proper weights */
  virtual void merge(const std::vector<double> &weights,
                     const std::vector<const Gaussian*> &gaussians,
                     bool finish_statistics = true);
  /// Compute the Kullback-Leibler divergence KL(current||g)
  virtual double kullback_leibler(Gaussian &g) const;
  /// Draw a random sample from this Gaussian
  virtual void draw_sample(Vector &sample);

  /** Tells if full statistics have been accumulated for this Gaussian
   * \param accum_pos Accumulator position
   */
  bool full_stats_accumulated(int accum_pos);

  /// Is this a diagonal covariance gaussian?
  virtual bool is_diagonal_covariance(void) const = 0;

  /**
   * \return true if Gaussian parameters are computed and valid
   */
  bool valid_parameters(void) const { return m_valid_parameters; }

  /** Apply ismoothing to the statistics.
   * \param source    Source buffer (the one being smoothed with)
   * \param target    Target buffer (the one the source buffer is added to)
   * \param smoothing Smoothing constant
   */
  void ismooth_statistics(int source, int target, double smoothing);

  double covariance_determinant() { return m_covariance_determinant; }
  
  // Accessing the accumulator
  double get_accumulated_gamma(int accum) const { return m_accums[accum]->gamma(); }
  void get_accumulated_mean(int accum, Vector &mean) const { m_accums[accum]->get_accumulated_mean(mean); }
  void get_accumulated_second_moment(int accum, Vector &s) { m_accums[accum]->get_accumulated_second_moment(s); }

  double get_accumulated_aux_gamma(int accum) { return m_accums[accum]->aux_gamma(); }


  // Special function to manipulate accumulators for ML smoothing
  void copy_gamma_to_aux_gamma(int source, int target);


protected:
  double m_constant;
  double m_covariance_determinant;
  bool m_valid_parameters;
  
  std::vector<GaussianAccumulator*> m_accums;

  Matrix *chol;

  double m_ebw_max_kld; //!< Not used if non-positive
  float m_fixed_d; //!< Not used if negative

  double m_min_d; //!< Filled with the minimum D after an EBW update
  double m_realized_d; //!< Filled with the realized D after an EBW update

  friend class HmmSet;
  friend class PDFPool;
  friend struct PDFPool::Gaussian_occ_comp;
};



class DiagonalGaussian : public Gaussian {
public:
  DiagonalGaussian(int dim);
  DiagonalGaussian(const DiagonalGaussian &g);
  ~DiagonalGaussian();
  virtual void reset(int dim);

  // From pdf
  virtual double compute_likelihood(const Vector &f) const;
  virtual double compute_log_likelihood(const Vector &f) const;
  virtual void write(std::ostream &os) const;
  virtual void read(std::istream &is);

  // Gaussian-specific
  virtual void start_accumulating(StatisticsMode mode);
  virtual void get_mean(Vector &mean) const;
  virtual void get_covariance(Matrix &covariance) const;
  virtual void set_mean(const Vector &mean);
  virtual void set_covariance(const Matrix &covariance,
                              bool finish_statistics = true);
  virtual Gaussian* copy_gaussian(void) { return new DiagonalGaussian(*this); }
  virtual void split(Gaussian &s1, Gaussian &s2, double perturbation = 0.2) const;
  virtual double compute_likelihood_exponential(const Vector &exponential_feature) const;
  virtual double compute_log_likelihood_exponential(const Vector &exponential_feature) const;

  // Faster implementations for DiagonalGaussian
  virtual void draw_sample(Vector &sample);
  virtual double kullback_leibler(Gaussian &g) const;

  /// Is this a diagonal covariance gaussian?
  virtual bool is_diagonal_covariance(void) const  { return true; }
  
  // Diagonal-specific
  /// Get the diagonal of the covariance matrix
  virtual void get_covariance(Vector &covariance) const;
  /// Set the diagonal of the covariance matrix
  virtual void set_covariance(const Vector &covariance,
                              bool finish_statistics = true);
#ifdef _CUDA
  /// Get the diagonal of the precision matrix
  virtual void get_precision(Vector &precision) const;
#endif

  /// Sets the constant after the precisions have been set
  void set_constant(void);

private:  
  Vector m_mean;
  Vector m_covariance;
  Vector m_precision;

  bool m_full_stats;
};



class FullCovarianceGaussian : public Gaussian {
public:
  FullCovarianceGaussian(int dim);
  FullCovarianceGaussian(const FullCovarianceGaussian &g);
  ~FullCovarianceGaussian();
  virtual void reset(int dim);

  // From pdf
  virtual double compute_likelihood(const Vector &f) const;
  virtual double compute_log_likelihood(const Vector &f) const;
  virtual void write(std::ostream &os) const;
  virtual void read(std::istream &is);

  // Gaussian-specific
  virtual void start_accumulating(StatisticsMode mode);
  virtual void get_mean(Vector &mean) const;
  virtual void get_covariance(Matrix &covariance) const;
  virtual void set_mean(const Vector &mean);
  virtual void set_covariance(const Matrix &covariance,
                              bool finish_statistics = true);
  virtual Gaussian* copy_gaussian(void) { return new FullCovarianceGaussian(*this); }
  virtual double compute_likelihood_exponential(const Vector &exponential_feature) const;
  virtual double compute_log_likelihood_exponential(const Vector &exponential_feature) const;

  /// Is this a diagonal covariance gaussian?
  virtual bool is_diagonal_covariance(void) const  { return false; }

  // Full-covariance-specific
  void recompute_exponential_parameters();
  
private:
  Vector m_mean;
  Matrix m_covariance;
  Matrix m_precision;
  Vector m_exponential_parameters;
  double m_exponential_normalizer;
};



#ifdef USE_SUBSPACE_COV
class PrecisionConstrainedGaussian : public Gaussian {
public:
  PrecisionConstrainedGaussian();
  PrecisionConstrainedGaussian(PrecisionSubspace *space);
  PrecisionConstrainedGaussian(const PrecisionConstrainedGaussian& g);
  ~PrecisionConstrainedGaussian();
  virtual void reset(int dim);

  // From pdf
  virtual double compute_likelihood(const Vector &f) const;
  virtual double compute_log_likelihood(const Vector &f) const;
  virtual void write(std::ostream &os) const;
  virtual void read(std::istream &is);

  // Gaussian-specific
  virtual void start_accumulating(StatisticsMode mode);
  virtual void get_mean(Vector &mean) const;
  virtual void get_covariance(Matrix &covariance) const;
  virtual void set_mean(const Vector &mean);
  virtual void set_covariance(const Matrix &covariance,
                              bool finish_statistics = true);
  virtual Gaussian* copy_gaussian(void) { return new PrecisionConstrainedGaussian(*this); }
  virtual double compute_likelihood_exponential(const Vector &exponential_feature) const;
  virtual double compute_log_likelihood_exponential(const Vector &exponential_feature) const;

  /// Is this a diagonal covariance gaussian?
  virtual bool is_diagonal_covariance(void) const  { return false; }

  // PCGMM-specific

  /* Get the transformed mean */
  void get_transformed_mean(Vector &transformed_mean) const { transformed_mean.copy(m_transformed_mean); }
  /* Set the transformed mean */
  void set_transformed_mean(const Vector &transformed_mean) { m_transformed_mean.copy(transformed_mean); }
  
  /* Get the coefficients for the subspace constrained precision matrix */
  void get_precision_coeffs(Vector &coeffs) const { coeffs.copy(m_coeffs); }
  /* Set the coefficients for the subspace constrained precision matrix */
  void set_precision_coeffs(const Vector &coeffs) { m_coeffs.copy(coeffs); }
  /* Get the subspace dimensionality */
  int subspace_dim() const { return m_coeffs.size(); }
  /* Get the subspace */
  PrecisionSubspace* get_subspace() const { return m_ps; }
  /* Set the subspace */
  void set_subspace(PrecisionSubspace *space) { m_ps = space; }
  /* Recomputes the m_constant */
  void recompute_constant();
  

private:
  Vector m_transformed_mean;
  Vector m_coeffs;
  PrecisionSubspace *m_ps;
};



class SubspaceConstrainedGaussian : public Gaussian {
public:
  SubspaceConstrainedGaussian();
  SubspaceConstrainedGaussian(ExponentialSubspace *space);
  SubspaceConstrainedGaussian(const SubspaceConstrainedGaussian &g);
  ~SubspaceConstrainedGaussian();
  virtual void reset(int feature_dim);

  // From pdf
  virtual double compute_likelihood(const Vector &f) const;
  virtual double compute_log_likelihood(const Vector &f) const;
  virtual void write(std::ostream &os) const;
  virtual void read(std::istream &is);

  // Gaussian-specific
  virtual void start_accumulating(StatisticsMode mode);
  virtual void get_mean(Vector &mean) const;
  virtual void get_covariance(Matrix &covariance) const;
  virtual void set_mean(const Vector &mean);
  virtual void set_covariance(const Matrix &covariance,
                              bool finish_statistics = true);
  virtual void set_parameters(const Vector &mean,
                              const Matrix &covariance);
  virtual Gaussian* copy_gaussian(void) { return new SubspaceConstrainedGaussian(*this); }
  virtual double compute_likelihood_exponential(const Vector &exponential_feature) const;
  virtual double compute_log_likelihood_exponential(const Vector &exponential_feature) const;

  /// Is this a diagonal covariance gaussian?
  virtual bool is_diagonal_covariance(void) const  { return false; }

  // SCGMM-specific

  /* Get the coefficients for the subspace constrained exponential parameters */
  void get_subspace_coeffs(Vector &coeffs) const { coeffs.copy(m_coeffs); }
  /* Set the coefficients for the subspace constrained exponential parameters */
  void set_subspace_coeffs(const Vector &coeffs) { m_coeffs.copy(coeffs); }
  /* Set the limited memory BFGS algorithm class for parameter optimization */
  void set_bfgs(HCL_UMin_lbfgs_d *bfgs) { m_bfgs = bfgs; }


  /* Get the subspace dimensionality */
  int subspace_dim() const { return m_coeffs.size(); }
  /* Get the subspace */
  ExponentialSubspace* get_subspace() const { return m_es; }
  /* Set the subspace */
  void set_subspace(ExponentialSubspace *space) { m_es = space; }
  
private:
  Vector m_coeffs;
  ExponentialSubspace *m_es;
  HCL_UMin_lbfgs_d *m_bfgs;
};
#endif


class Mixture : public PDF {
public:
  // Mixture-specific
  Mixture();
  Mixture(PDFPool *pool);
  ~Mixture();
  int size() const { return m_pointers.size(); };
  void reset();
  void set_pool(PDFPool *pool);
  PDFPool* get_pool() { return m_pool; }
  /** Set the mixture components, clear existing mixture */
  void set_components(const std::vector<int> &pointers,
		      const std::vector<double> &weights);
  /** \return the mixture component */
  PDF* get_base_pdf(int index);

  /** \return a pool index for a mixture component */
  int get_base_pdf_index(int index) { return m_pointers[index]; }

  /** Get all the mixture components */
  void get_components(std::vector<int> &pointers,
		      std::vector<double> &weights);
  
  /** Add one new component to the mixture. 
   * Doesn't normalize the coefficients in between */
  void add_component(int pool_index, double weight);
  
  /** Normalize the weights to have a sum of 1 */
  void normalize_weights();
  
  /** Changes the mixture coefficient for a mixture component*/
  void set_mixture_coefficient(int index, double coeff) { m_weights[index] = coeff; }
  
  /** \return the mixture coefficient for a mixture component*/
  double get_mixture_coefficient(int index) const { return m_weights[index]; }

  /** Returns the relative component index of a PDF
   * \param p PDF pool index
   * \return Index of the component in this mixture, -1 if not found
   */
  int component_index(int p);

  /** Updates component indices and deletes removed components.
   * Normalizes the weights after component deletion.
   * \param cmap Reference to a vector which maps old indices to new ones
   *             The deleted components are marked as -1.
   */
  void update_components(const std::vector<int> &cmap);

  /** Deletes one component from the mixture and normalizes the weights
   * \param index Component index
   */
  void remove_component(int index);

  // For accessing the accumulator
  double get_accumulated_gamma(int accum, int index) { return m_accums[accum]->gamma[index]; }

  /** Computes the Cross-entropy between this and another mixture
   * using Monte Carlo simulation and sampling from the current distribution
   * \param g the other mixture
   * \param samples number of samples to use in the computation
   */
  double cross_entropy(Mixture &g, int samples=10000);
  
  /** Computes the Kullback-Leibler divergence between this and another mixture
   * using Monte Carlo simulation and sampling from the current distribution
   * \param g the other mixture
   * \param samples number of samples to use in the mc-simulation
   */
  double kullback_leibler(Mixture &g, int samples=10000);

  /** Computes the Kullback-Leibler divergence between this and another mixture
   * using sampling with the (f+g)/2 as sampling distribution
   * \param g the other mixture
   * \param samples number of samples to use in the computation
   */
  double bhattacharyya(Mixture &g, int samples=10000);

  virtual void accumulate_aux_gamma(double gamma, int accum_pos = 0);
  double get_accumulated_mixture_ll(int accum) { return m_accums[accum]->mixture_ll; }
  double get_accumulated_aux_gamma(int accum) { return m_accums[accum]->aux_gamma; }

  // Special function to manipulate accumulators for ML smoothing
  void copy_aux_gamma(int source, int target);
  
  // From pdf
  virtual void start_accumulating(StatisticsMode mode);
  virtual bool is_accumulating() const { return (m_accums.size()>0?true:false); }
  virtual void accumulate(double prior,
			  const Vector &f,
			  int accum_pos = 0);
  virtual void dump_statistics(std::ostream &os) const;
  virtual void accumulate_from_dump(std::istream &is, StatisticsMode mode);
  virtual void stop_accumulating();
  virtual bool accumulated(int accum_pos = 0) const;
  virtual void estimate_parameters(EstimationMode mode);
  virtual double compute_likelihood(const Vector &f) const;
  virtual double compute_log_likelihood(const Vector &f) const;
  virtual void write(std::ostream &os) const;
  virtual void read(std::istream &is);
  virtual void draw_sample(Vector &sample);

private:

  class MixtureAccumulator {
  public:
    inline MixtureAccumulator(int mixture_size);
    std::vector<double> gamma;
    double aux_gamma;
    double mixture_ll;
    
    bool accumulated;
  };

  std::vector<int> m_pointers;
  std::vector<double> m_weights;

  PDFPool *m_pool;
  std::vector<MixtureAccumulator*> m_accums;
};


Mixture::MixtureAccumulator::MixtureAccumulator(int mixture_size) {
  gamma.resize(mixture_size); 
  for (int i=0; i<mixture_size; i++)
    gamma[i]=0.0;
  aux_gamma = 0;
  mixture_ll = 0;
  accumulated = false;
}

// Comparison function to get a sorted list of Gaussians with decreasing
// occupancies
struct PDFPool::Gaussian_occ_comp
{
  Gaussian_occ_comp(std::vector<PDF*> &pool) : m_pool(pool) { }
  bool operator()(int x, int y) { return dynamic_cast< Gaussian* > (m_pool[x]) ->m_accums[PDF::ML_BUF]->gamma() > dynamic_cast< Gaussian* > (m_pool[y]) ->m_accums[PDF::ML_BUF]->gamma(); }
private:
  std::vector<PDF*> &m_pool;
};

}

#endif /* DISTRIBUTIONS_HH */
