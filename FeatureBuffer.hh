#ifndef FEATUREBUFFER_HH
#define FEATUREBUFFER_HH

#include <string>
#include <assert.h>
#include <vector>

class FeatureVec {
public:
  FeatureVec() : m_ptr(NULL), m_dim(0) { }
  FeatureVec(const float *ptr, int dim) : m_ptr(ptr), m_dim(dim) { }

  const float &operator[](int index) const 
  { 
    if (index < 0 || index >= m_dim)
      throw std::string("FeatureVec out of bounds");
    return m_ptr[index]; 
  }

  float &operator[](int index) 
  { 
    if (index < 0 || index >= m_dim)
      throw std::string("FeatureVec out of bounds");
    return ((float *)m_ptr)[index];
  }

  int dim() const { return m_dim; }

private:
  const float *m_ptr;
  int m_dim;
};

/** A class for storing feature vectors in a circular window buffer. */
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
    int index = modulo(frame, m_num_frames);
    return FeatureVec(&m_buffer[index * m_dim], m_dim);
  }

  /** Mutable access to the values in the buffer. */
  FeatureVec operator[](int frame) 
  { 
    int index = modulo(frame, m_num_frames);
    return FeatureVec(&m_buffer[index * m_dim], m_dim);
  }

private:

  /** Compute modulo of two values so that negative arguments are
   * handled correctly. */
  int modulo(int a, int b) const
  {
    int result = a % b;
    if (result < 0)
      result += b;
    return result;
  }

  int m_dim; //!< The dimension of the feature vectors
  int m_num_frames; //!< The total number of frames in the window
  std::vector<float> m_buffer; //!< The actual data stored in the buffer
};


#endif /* FEATUREBUFFER_HH */
