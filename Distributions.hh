#ifndef DISTRIBUTIONS_HH
#define DISTRIBUTIONS_HH

#include "gmd.h"
#include "lavd.h"
#include "FeatureBuffer.hh"
#include "FeatureModules.hh"
#include "LinearAlgebra.hh"



class FullStatisticsAccumulator {
public:
  FullStatisticsAccumulator(int dim) { 
    mean.resize(dim);
    cov.resize(dim,dim);
    mean=0;
    cov=0;
    gamma=0;
    accumulated=false;
  };
  double gamma;
  Vector mean;
  Matrix cov;
  bool accumulated;
};


class DiagonalStatisticsAccumulator {
public:
  DiagonalStatisticsAccumulator(int dim) { 
    mean.resize(dim);
    cov.resize(dim);
    mean=0;
    cov=0;
    gamma=0;
    accumulated=false;
  };
  bool accumulated;
  double gamma;
  Vector mean;
  Vector cov;
};



class PDF {
public:

  enum EstimationMode { ML, MMI };
  
  // COMMON

  virtual ~PDF() {}
  /* The feature dimensionality */
  int dim() const { return m_dim; }
  
  // TRAINING

  /* Initializes the accumulator buffers */  
  virtual void start_accumulating() = 0;
  /* Accumulates the statistics for this pdf */
  virtual void accumulate(double prior,
			  const FeatureVec &f, 
			  int accum_pos = 0) = 0;
  /* Writes the currently accumulated statistics to a file */
  virtual void dump_statistics(std::ostream &os,
			       int accum_pos = 0) const = 0;
  /* Accumulates from a file dump */
  virtual void accumulate_from_dump(std::istream &is) = 0;
  /* Stops training and clears the accumulators */
  virtual void stop_accumulating() = 0;
  /* Tells if this pdf has been accumulated */
  virtual bool accumulated(int accum_pos = 0) const = 0;
  /* Use the accumulated statistics to update the current model parameters. */
  virtual void estimate_parameters() = 0;
  /* Sets the training mode for this pdf */
  void set_estimation_mode(EstimationMode mode) { m_mode = mode; }
  /* Sets the training mode for this pdf */
  EstimationMode estimation_mode() const { return m_mode; }
  /* Returns the correct accumulator for these statistics */
  int accumulator_position(std::string type);
  /* Sets the constant D for MMI updates */
  void set_mmi_d_constant(double d_constant) { m_d_constant = d_constant; }
  
  
  // LIKELIHOODS
  
  /* The likelihood of the current feature given this model */
  virtual double compute_likelihood(const FeatureVec &f) const = 0;
  /* The log likelihood of the current feature given this model */
  virtual double compute_log_likelihood(const FeatureVec &f) const = 0;

  // IO

  /* Write the parameters of this distribution to the stream os */
  virtual void write(std::ostream &os) const = 0;
  /* Read the parameters of this distribution from the stream is */
  virtual void read(std::istream &is) = 0;


protected:
  int m_dim;
  double m_d_constant;
  EstimationMode m_mode;
};




class PDFPool {
public:
  
  PDFPool();
  PDFPool(int dim);
  ~PDFPool();
  /* The dimensionality of the distributions in this pool */
  int dim() const { return m_dim; }
  /* The dimensionality of the distributions in this pool */
  int size() const { return m_pool.size(); };
  /* Reset everything */
  void reset();
  
  /* Get the pdf from the position index */
  PDF* get_pdf(int index) const;
  /* Set the pdf in the position index */
  void set_pdf(int index, PDF *pdf);
  
  /* Read the distributions from a .gk -file */
  void read_gk(const std::string &filename);
  /* Write the distributions to a .gk -file */
  void write_gk(const std::string &filename) const;
  
  /* Reset the cache */
  void reset_cache();

  /** Compute the likelihood of a feature for pdf in the pool. Uses cache.
   * \param f the feature vector
   * \param index the pdf index
   * \return the likelihood of the given feature for some pdf
   */
  double compute_likelihood(const FeatureVec &f, int index);

  /** Computes likelihoods for all distributions to the cache
   * \param f the feature vector
   */
  void precompute_likelihoods(const FeatureVec &f);
  
private:
  std::vector<PDF*> m_pool;
  std::vector<double> m_likelihoods;
  std::vector<int> m_valid_likelihoods;
  int m_dim;
};




class Gaussian : public PDF {
public:
  
  // ABSTRACT FUNCTIONS, SHOULD BE OVERWRITTEN IN THE GAUSSIAN IMPLEMENTATIONS
  
  /* Resets the Gaussian to have dimensionality dim and all values zeroed */
  virtual void reset(int dim) = 0;
  
  // FROM PDF

  /* Initializes the accumulator buffers */  
  virtual void start_accumulating() = 0;
  /* Accumulates the maximum likelihood statistics for the Gaussian
     weighed with a prior. */
  virtual void accumulate(double prior, const FeatureVec &f, 
			  int accum_pos = 0) = 0;
  /* Writes the currently accumulated statistics to a stream */
  virtual void dump_statistics(std::ostream &os,
			       int accum_pos = 0) const = 0;
  /* Accumulates from dump */
  virtual void accumulate_from_dump(std::istream &is) = 0;
  /* Stops training and clears the accumulators */
  virtual void stop_accumulating() = 0;
  /* Tells if this Gaussian has been accumulated */
  virtual bool accumulated(int accum_pos = 0) const = 0;
  /* Use the accumulated statistics to update the current model parameters. */
  virtual void estimate_parameters() = 0;
  
  // GAUSSIAN SPECIFIC
  
  /* Returns the mean vector for this Gaussian */
  virtual void get_mean(Vector &mean) const = 0;
  /* Returns the covariance matrix for this Gaussian */
  virtual void get_covariance(Matrix &covariance) const = 0;
  /* Sets the mean vector for this Gaussian */
  virtual void set_mean(const Vector &mean) = 0;
  /* Sets the covariance matrix for this Gaussian */
  virtual void set_covariance(const Matrix &covariance) = 0;
  
  // THESE FUNCTIONS HAVE ALSO A COMMON IMPLEMENTATION, BUT CAN BE OVERWRITTEN

  /* Splits the current Gaussian to two by disturbing the mean */
  virtual void split(Gaussian &s1, Gaussian &s2) const;
  /* Sets the parameters for the current Gaussian by merging m1 and m2 */
  virtual void merge(double w1, const Gaussian &m1, double w2, const Gaussian &m2);
  /* Compute the Kullback-Leibler divergence KL(current||g) */
  virtual double kullback_leibler(Gaussian &g) const;

  // EASY, NO NEED FOR OVERWRITING

  /* Set the minimum variance for this Gaussian */
  void set_minvar(double minvar) { m_minvar = minvar; };
  /* Set the minimum eigenvalue for this Gaussian */
  void set_mineig(double mineig) { m_mineig = mineig; };
  /* Set the covariance smoothing for this Gaussian */
  void set_covsmooth(double covsmooth) { m_covsmooth = covsmooth; };
  
private:
  double m_minvar;
  double m_mineig;
  double m_covsmooth;
};




class DiagonalGaussian : public Gaussian {
public:
  DiagonalGaussian(int dim);
  DiagonalGaussian(const DiagonalGaussian &g);
  ~DiagonalGaussian();
  virtual void reset(int dim);

  // From pdf
  virtual double compute_likelihood(const FeatureVec &f) const;
  virtual double compute_log_likelihood(const FeatureVec &f) const;
  virtual void write(std::ostream &os) const;
  virtual void read(std::istream &is);

  // Gaussian-specific
  virtual void start_accumulating();
  virtual void accumulate(double prior, const FeatureVec &f, 
			  int accum_pos = 0);
  virtual void dump_statistics(std::ostream &os,
			       int accum_pos = 0) const;
  virtual void accumulate_from_dump(std::istream &is);
  virtual void stop_accumulating();
  virtual bool accumulated(int accum_pos = 0) const;
  virtual void estimate_parameters();
  virtual void get_mean(Vector &mean) const;
  virtual void get_covariance(Matrix &covariance) const;
  virtual void set_mean(const Vector &mean);
  virtual void set_covariance(const Matrix &covariance);

  // Diagonal-specific
  /* Get the diagonal of the covariance matrix */
  void get_covariance(Vector &covariance) const;
  /* Set the diagonal of the covariance matrix */
  void set_covariance(const Vector &covariance);

private:  
  double m_constant;
  Vector m_mean;
  Vector m_covariance;
  Vector m_precision;
  
  std::vector<DiagonalStatisticsAccumulator*> m_accums;
};




class MlltGaussian : public Gaussian {
public:
  MlltGaussian(int dim);
  MlltGaussian(const DiagonalGaussian &g);
  MlltGaussian(const MlltGaussian &g);
  ~MlltGaussian();
  virtual void reset(int dim);

  // From pdf
  virtual double compute_likelihood(const FeatureVec &f) const;
  virtual double compute_log_likelihood(const FeatureVec &f) const;
  virtual void write(std::ostream &os) const;
  virtual void read(std::istream &is);

  // Gaussian-specific
  virtual void start_accumulating();
  virtual void accumulate(double prior, const FeatureVec &f, 
			  int accum_pos = 0);
  virtual void dump_statistics(std::ostream &os,
			       int accum_pos = 0) const;
  virtual void accumulate_from_dump(std::istream &is);
  virtual void stop_accumulating();
  virtual bool accumulated(int accum_pos = 0) const;
  virtual void estimate_parameters();
  virtual void get_mean(Vector &mean) const;
  virtual void get_covariance(Matrix &covariance) const;
  virtual void set_mean(const Vector &mean);
  virtual void set_covariance(const Matrix &covariance);

  // Diagonal-specific
  /* Get the diagonal of the covariance matrix */
  void get_covariance(Vector &covariance) const;
  /* Set the diagonal of the covariance matrix */
  void set_covariance(const Vector &covariance);

  // Mllt-specific
  void transform_parameters(LinTransformModule &A);
  static void update_mllt_transform(PDFPool &mllt_gaussians);
    
private:  
  double m_constant;
  Vector m_mean;
  Vector m_covariance;
  Vector m_precision;

  std::vector<FullStatisticsAccumulator*> m_accums;
};



class FullCovarianceGaussian : public Gaussian {
public:
  FullCovarianceGaussian(int dim);
  FullCovarianceGaussian(const FullCovarianceGaussian &g);
  ~FullCovarianceGaussian();
  virtual void reset(int dim);

  // From pdf
  virtual double compute_likelihood(const FeatureVec &f) const;
  virtual double compute_log_likelihood(const FeatureVec &f) const;
  virtual void write(std::ostream &os) const;
  virtual void read(std::istream &is);

  // Gaussian-specific
  virtual void start_accumulating();
  virtual void accumulate(double prior,
			  const FeatureVec &f, 
			  int accum_pos = 0);
  virtual void dump_statistics(std::ostream &os,
			       int accum_pos = 0) const;
  virtual void accumulate_from_dump(std::istream &is);
  virtual void stop_accumulating();
  virtual bool accumulated(int accum_pos = 0) const;
  virtual void estimate_parameters();
  virtual void get_mean(Vector &mean) const;
  virtual void get_covariance(Matrix &covariance) const;
  virtual void set_mean(const Vector &mean);
  virtual void set_covariance(const Matrix &covariance);
  
private:
  double m_constant;
  Vector m_mean;
  Matrix m_covariance;
  Matrix m_precision;
  
  std::vector<FullStatisticsAccumulator*> m_accums;
};




class Mixture : public PDF {
public:
  // Mixture-specific
  Mixture();
  Mixture(PDFPool *pool);
  ~Mixture();
  int size() const { return m_pointers.size(); };
  void reset();
  void set_pool(PDFPool *pool);
  /* Set the mixture components, clear existing mixture */
  void set_components(const std::vector<int> &pointers,
		      const std::vector<double> &weights);
  /* Get a mixture component */
  PDF* get_base_pdf(int index);
  /* Get all the mixture components */
  void get_components(std::vector<int> &pointers,
		      std::vector<double> &weights);
  /* Add one new component to the mixture. 
     Doesn't normalize the coefficients in between */
  void add_component(int pool_index, double weight);
  /* Normalize the weights to have a sum of 1 */
  void normalize_weights();

  // From pdf
  virtual void start_accumulating();
  virtual void accumulate(double prior,
			  const FeatureVec &f,
			  int accum_pos = 0);
  virtual void dump_statistics(std::ostream &os,
			       int accum_pos = 0) const;
  virtual void accumulate_from_dump(std::istream &is);
  virtual void stop_accumulating();
  virtual bool accumulated(int accum_pos = 0) const;
  virtual void estimate_parameters();
  virtual double compute_likelihood(const FeatureVec &f) const;
  virtual double compute_log_likelihood(const FeatureVec &f) const;
  virtual void write(std::ostream &os) const;
  virtual void read(std::istream &is);

private:

  class MixtureAccumulator {
  public:
    inline MixtureAccumulator(int mixture_size);
    std::vector<double> gamma;
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
  accumulated = false;
}

#endif /* DISTRIBUTIONS_HH */
