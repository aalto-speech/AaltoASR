
class PDF {
public:
  virtual double compute_likelihood(const FeatureVec &f) const = 0;
  virtual double compute_log_likelihood(const FeatureVec &f) const = 0;
  virtual void write(std::ostream &os) const = 0;
  virtual void read(std::istream &is) = 0;
}


class Gaussian : public PDF {
public:

  enum EstimationMode { ML, MMI };

  // Abstract virtual
  virtual void accumulate_ml(double prior, const FeatureVec &f) = 0;
  virtual void accumulate_mmi_denominator(std::vector<double> priors,
					  std::vector<const FeatureVec*> const features);
  virtual void estimate_parameters() = 0;
  virtual vector &get_mean() const = 0;
  virtual matrix &get_covariance() const = 0;
  virtual void set_mean(vector &mean) = 0;
  virtual void set_covariance(matrix &covariance) = 0;
  
  // Can be overwritten if needed
  virtual void split(Gaussian &s1, Gaussian &s2) const;
  virtual void merge(Gaussian &m1, Gaussian &m2);
  virtual void merge(Gaussian &m);
  virtual double kullback_leibler(Gaussian &g) const;

private:
  Gaussian::EstimationMode;
}


class DiagonalCovarianceGaussian : public Gaussian {
public:
  // From pdf
  virtual double compute_likelihood(const FeatureVec &f);
  virtual double compute_log_likelihood(const FeatureVec &f);
  virtual void write(std::ostream &os);
  virtual void read(std::istream &is);

  // Gaussian-specific
  virtual void accumulate_ml(double prior, const FeatureVec &f);
  virtual void accumulate_mmi_denominator(std::vector<double> priors,
					  std::vector<const FeatureVec*> const features);
  virtual double accumulate(double prior, const FeatureVec &f) = 0;
  virtual void estimate_parameters() = 0;
  virtual vector &get_mean();
  virtual matrix &get_covariance();
  virtual void set_mean(vector &mean);
  virtual void set_covariance(matrix &covariance);

  // Diagonal-specific
  vector &get_covariance();
  void set_covariance(vector &covariance);

private:
  vector mean;
  vector covariance;
}


class FullCovarianceGaussian : public Gaussian {
public:
  // From pdf
  virtual double compute_likelihood(const FeatureVec &f);
  virtual double compute_log_likelihood(const FeatureVec &f);
  virtual void write(std::ostream &os);
  virtual void read(std::istream &is);

  // Gaussian-specific
  virtual void accumulate_ml(double prior, const FeatureVec &f);
  virtual void accumulate_mmi_denominator(std::vector<double> priors,
					  std::vector<const FeatureVec*> const features);
  virtual void estimate_parameters();
  virtual vector &get_mean();
  virtual matrix &get_covariance();
  virtual void set_mean(vector &mean) = 0;
  virtual void set_covariance(matrix &covariance) = 0;
  
private:
  vector mean;
  matrix covariance;
}


class PrecisionConstrainedGaussian : public Gaussian {
public:
  // From pdf
  double compute_likelihood(const FeatureVec &f);
  double compute_log_likelihood(const FeatureVec &f);
  void write(std::ostream &os);
  void read(std::istream &is);
  
  // Gaussian-specific
  virtual void accumulate_ml(double prior, const FeatureVec &f);
  virtual void accumulate_mmi_denominator(std::vector<double> priors,
					  std::vector<const FeatureVec*> const features);
  virtual void estimate_parameters();
  virtual vector &get_mean();
  virtual matrix &get_covariance();
  virtual void set_mean(vector &mean);
  virtual void set_covariance(matrix &covariance);

  // PCGMM-specific
  vector &get_precision_coeffs();
  void set_precision_coeffs(vector &coeffs);

private:
  vector mean;
  vector precision_coeffs;
  PrecisionSubspace &ps;
}


class SubspaceConstrainedGaussian : public Gaussian {
public:
  // From pdf
  virtual double compute_likelihood(const FeatureVec &f);
  virtual double compute_log_likelihood(const FeatureVec &f);
  virtual void write(std::ostream &os);
  virtual void read(std::istream &is);

  // Gaussian-specific
  virtual void accumulate_ml(double prior, const FeatureVec &f);
  virtual void accumulate_mmi_denominator(std::vector<double> priors const,
					  std::vector<const FeatureVec*> const features);
  virtual void estimate_parameters();
  virtual vector &get_mean();
  virtual matrix &get_covariance;
  virtual void set_mean(vector &mean);
  virtual void set_covariance(matrix &covariance);

  // SCGMM-specific
  vector &get_subspace_coeffs();
  void set_subspace_coeffs(vector &coeffs);

private:
  vector subspace_coeffs;
  ExponentialSubspace &es;
}


class Mixture : public PDF {
public:
  virtual double compute_likelihood(const FeatureVec &f);
  virtual double compute_log_likelihood(const FeatureVec &f);
  virtual void write(std::ostream &os);
  virtual void read(const std::istream &is);
private:
  vector mixture_weights;
  vector mixture_pointers;
  PDFPool &pp;
}


class PDFPool {
public:
  PDF &get_pdf(int pdfindex);
  void set_pdf(int pdfindex, PDF &pdf);
private:
  std::vector<PDF> pool;
}
