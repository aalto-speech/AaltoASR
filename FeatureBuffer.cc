#include "FeatureBuffer.hh"

FeatureBuffer::FeatureBuffer()
  : m_start_frame(0),
    m_end_frame(0),
    m_dim(0),
    m_features(0),
    m_growing_allowed(false)
{
}

FeatureBuffer::FeatureBuffer(int start_frame, int end_frame, int dim)
{
  move(start_frame, end_frame, dim);
}

// NOTES:
// - dim = 0 means "keep old dim"
// - After dim is set, it can not be changed (FIXME)
// INVARIANTS:
// - start of frame f is always stored at position 
//   modulo(f, m_features) * m_dim
void
FeatureBuffer::move(int start_frame, int end_frame, int dim)
{
  // Change and check dimension
  if (dim != 0) {
    if (m_dim != 0 && dim != m_dim)
      throw DimensionChanged();
    m_dim = dim;
  }

  // Allocate new space if necessary
  int features = end_frame - start_frame;
  if (features > m_features) {
    if (!m_growing_allowed && m_features > 0)
      throw GrowingNotAllowed();

    std::vector<float> m_new_buffer;
    m_new_buffer.resize(features * m_dim);

    // Copy common data
    int copy_start = start_frame > m_start_frame ? start_frame : m_start_frame;
    int copy_end = end_frame < m_end_frame ? end_frame : m_end_frame;
    for (int f = copy_start; f < copy_end; f++) {
      for (int d = 0; d < m_dim; d++) {
	m_new_buffer[modulo(f, features) * m_dim + d] = 
	  m_feature_buffer[modulo(f, m_features) * m_dim + d];
      }
    }
    m_feature_buffer.swap(m_new_buffer);
    m_features = features;
  }

  // Update pointers
  m_start_frame = start_frame;
  m_end_frame = end_frame;
}

