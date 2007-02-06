#ifndef FEATUREBUFFER_HH
#define FEATUREBUFFER_HH

#include <string>
#include <assert.h>
#include <vector>
#include "util.hh"

/** Class for accessing feature vectors of the \ref FeatureBuffer
 * class with proper const and array-bounds checking. */
class FeatureVec {
public:
  /** Default constructor. */
  FeatureVec() : m_ptr(NULL), m_dim(0) { }

  /** Construct a vector. 
   *
   * \param ptr = pointer to the feature vector array
   * \param dim = the dimension of the vector
   */
  FeatureVec(const float *ptr, int dim) : m_ptr(ptr), m_dim(dim) { }

  /** Copy the contents of a feature vector */
  void copy(const FeatureVec &vec)
  {
    assert(vec.dim() == m_dim);
    for (int i = 0; i < m_dim; i++)
      const_cast<float*>(m_ptr)[i] = vec[i];
  }

  /** Set the feature values from a std::vector. */
  void set(const std::vector<float> &vec)
  {
    assert((int)vec.size() == m_dim);
    for (int i = 0; i < m_dim; i++)
      const_cast<float*>(m_ptr)[i] = vec[i];
  }

  /** Fill std::vector with the feature values. */
  void get(std::vector<float> &vec) const
  {
    vec.resize(m_dim);
    for (int i = 0; i < m_dim; i++)
      vec[i] = m_ptr[i];
  }

  /** Constant access to feature vector values. */
  const float &operator[](int index) const 
  { 
    if (index < 0 || index >= m_dim)
      throw std::string("FeatureVec out of bounds");
    return m_ptr[index]; 
  }

  /** Access to feature vector values. */
  float &operator[](int index) 
  { 
    if (index < 0 || index >= m_dim)
      throw std::string("FeatureVec out of bounds");
    return (const_cast<float*>(m_ptr))[index];
  }

  /** The dimension of the vector. */
  int dim() const { return m_dim; }

private:
  const float *m_ptr; //!< Pointer to the feature vector values
  int m_dim; //!< The dimensino of the vector
};

/** A class for storing feature vectors in a circular buffer. */
class FeatureBuffer {
public:

  /** Default constructor */
  FeatureBuffer() : m_dim(1), m_num_frames(1), m_buffer(1) { }

  /** Set the size of the window.  \warning Invalidates all data in
   * the buffer. */
  void resize(int num_frames, int dim)
  {
    assert(num_frames > 0);
    assert(dim > 0);
    m_num_frames = num_frames;
    m_dim = dim;
    m_buffer.resize(m_num_frames * m_dim);
  }

  /** The dimension of the feature vectors. */
  int dim() const { return m_dim; }

  /** The number of frames in the buffer. */
  int num_frames() const { return m_num_frames; }

  /** Constant access to the values in the buffer. */
  const FeatureVec operator[](int frame) const 
  { 
    int index = util::modulo(frame, m_num_frames);
    return FeatureVec(&m_buffer[index * m_dim], m_dim);
  }

  /** Mutable access to the values in the buffer. */
  FeatureVec operator[](int frame) 
  { 
    int index = util::modulo(frame, m_num_frames);
    return FeatureVec(&m_buffer[index * m_dim], m_dim);
  }

private:

  int m_dim; //!< The dimension of the feature vectors
  int m_num_frames; //!< The total number of frames in the window
  std::vector<float> m_buffer; //!< The actual data stored in the buffer
};


#endif /* FEATUREBUFFER_HH */
