#include "FeatureModules.hh"
#include "FeatureGenerator.hh"

#include <math.h>
#include "util.hh"


FeatureModule::FeatureModule() :
  m_own_offset_left(-1),
  m_own_offset_right(-1),
  m_req_offset_left(0),
  m_req_offset_right(0),
  m_init_offset_left(0),
  m_init_offset_right(0),
  m_buffer_size(0),
  m_buffer_last_pos(INT_MAX),
  m_dim(0)
{
}

FeatureModule::~FeatureModule()
{
}

void
FeatureModule::set_buffer(int left, int right)
{
  int new_size;
  
  assert( left >= 0 );
  assert( right >= 0 );
  assert( m_own_offset_left >= 0 );
  assert( m_own_offset_right >= 0 );
  
  if (left > m_req_offset_left)
    m_req_offset_left = left;
  if (right > m_req_offset_right)
    m_req_offset_right = right;
  new_size = m_req_offset_right + m_req_offset_left + 1;
  m_buffer_last_pos = INT_MAX; // Invalidate the buffer
  if (new_size > m_buffer_size)
  {
    m_buffer_size = new_size;
    assert( m_buffer_size > 0 );
    m_buffer.resize(m_buffer_size, m_dim);
    if (m_own_offset_left+m_own_offset_right > 0)
    {
      // Require buffering from source modules
      for (int i = 0; i < (int)m_sources.size(); i++)
        m_sources[i]->set_buffer(m_own_offset_left, m_own_offset_right);
    }
  }
}

void
FeatureModule::update_init_offsets(const FeatureModule &target)
{
  int left = target.m_init_offset_left + target.m_own_offset_left;
  int right = target.m_init_offset_right + target.m_own_offset_right;

  if (left > m_init_offset_left)
    m_init_offset_left = left;
  if (right > m_init_offset_right)
    m_init_offset_right = right;

  assert(m_init_offset_left >= m_req_offset_left);
  assert(m_init_offset_right >= m_req_offset_right);
}

void
FeatureModule::require_init_buffer(void)
{  
  int new_size = m_init_offset_left + m_init_offset_right + 1;
  
  if (new_size > m_buffer_size)
  {
    m_buffer_size = new_size;
    m_buffer.resize(m_buffer_size, m_dim);
  }
}


const FeatureVec
FeatureModule::at(int frame)
{
  int buffer_gen_start;
  
  if (frame <= m_buffer_last_pos &&
      frame > m_buffer_last_pos - m_buffer_size)
    return m_buffer[frame];

  if (frame > m_buffer_last_pos)
  {
    // Moving forward, reuse the buffer if possible
    buffer_gen_start = m_buffer_last_pos + 1;
    if (frame >= buffer_gen_start + m_buffer_size)
      buffer_gen_start = frame - m_buffer_size + 1;
  }
  else
  {
    // Moving backwards, recompute the entire buffer
    buffer_gen_start = frame - m_buffer_size + 1;
  }
  m_buffer_last_pos = frame;
  
  // Generate the buffer
  for (int i = buffer_gen_start; i <= m_buffer_last_pos; i++)
    generate(i);
  return m_buffer[frame];
}


// The default implementation allows only one source, overload if necessary.
void
FeatureModule::add_source(FeatureModule *source)
{
  if (m_sources.size() > 0)
  {
    throw std::string("Multiple sources are not allowed for module ") +
      m_type_str;
  }
  m_sources.push_back(source);
}

void
FeatureModule::get_config(ModuleConfig &config)
{
  config.set("name", m_name);
  config.set("type", m_type_str);
  get_module_config(config);
}

void
FeatureModule::set_config(const ModuleConfig &config)
{
  set_module_config(config);
  assert( m_own_offset_left >= 0 );
  assert( m_own_offset_right >= 0 );
  assert( m_dim > 0 );

  // Initialize own buffer and propagate requests to sources if necessary
  set_buffer(0, 0);
}

void
FeatureModule::reset()
{
  m_buffer_last_pos = INT_MAX;
  reset_module();
}

void
FeatureModule::print_dot_node(FILE *file)
{
  fprintf(file, "  %s [label=\""
          "%s\\n"
          "own=%d-%d\\n"
          "req=%d-%d\\n"
          "init=%d-%d\\n"
          "buf=%d\\n"
          "\"]\n", 
          m_name.c_str(), m_name.c_str(),
          m_own_offset_left, m_own_offset_right,
          m_req_offset_left, m_req_offset_right,
          m_init_offset_left, m_init_offset_right,
          m_buffer_size
    );
}

//////////////////////////////////////////////////////////////////
// FFTModule
//////////////////////////////////////////////////////////////////

FFTModule::FFTModule(FeatureGenerator *fea_gen) :
  m_fea_gen(fea_gen),
  m_sample_rate(0),
  m_frame_rate(0),
  m_window_advance(0),
  m_window_width(0),
  m_eof_frame(INT_MAX),
  m_coeffs(NULL),
  m_copy_borders(true),
  m_last_feature_frame(INT_MIN)
{
  m_type_str = type_str();
}


FFTModule::~FFTModule()
{
  if (m_coeffs)
    fftw_destroy_plan(m_coeffs);
  discard_file();
}


void
FFTModule::set_file(FILE *fp)
{
  if (m_fea_gen->audio_format() == FeatureGenerator::AF_RAW)
  {
    m_reader.open_raw(fp, m_sample_rate);
  }
  else if (m_fea_gen->audio_format() == FeatureGenerator::AF_AUTO)
  {
    m_reader.open(fp);
  }
  else
  {
    throw std::string("Trying to open an unknown file");
  }
  // Check that sample rate matches that given in configuration
  if (m_reader.sample_rate() != m_sample_rate)
  {
    throw std::string(
      "File sample rate does not match the model configuration");
  }
  m_eof_frame = INT_MAX; // No EOF frame encountered yet
}


void
FFTModule::discard_file(void)
{
  m_reader.close();
}


bool
FFTModule::eof(int frame)
{
  if (frame < m_eof_frame)
    return false;
  return true;
}

int FFTModule::last_frame(void)
{
  // FIXME: NOT TESTED!!!
  return (int)((m_reader.num_samples()-m_window_width-1)/m_window_advance);
}

void
FFTModule::get_module_config(ModuleConfig &config)
{
  assert(m_sample_rate > 0);
  config.set("sample_rate", m_sample_rate);
  config.set("frame_rate", m_frame_rate);
  config.set("window_width", m_window_width);
  config.set("copy_borders", m_copy_borders);
  config.set("pre_emph_coef", m_emph_coef);
  config.set("magnitude", m_magnitude);
  if (m_log)
    config.set("log", m_log);
}

void
FFTModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;
  
  if (!config.get("sample_rate", m_sample_rate))
    throw std::string("FFTModule: Must set sample rate");

  m_frame_rate = 125;
  config.get("frame_rate", m_frame_rate);
  m_window_advance = m_sample_rate/m_frame_rate;
  m_window_width = (int)(2*m_sample_rate/m_frame_rate);
  config.get("window_width", m_window_width);

  m_copy_borders = 1;
  config.get("copy_borders", m_copy_borders);

  m_emph_coef = 0.97;
  config.get("pre_emph_coef", m_emph_coef);
  m_magnitude = 1;
  config.get("magnitude", m_magnitude);
  m_log = 0;
  config.get("log", m_log);

  m_dim = m_window_width/2+1;
  m_hamming_window.resize(m_window_width);
  for (int i = 0; i < m_window_width; i++)
    m_hamming_window[i] = .54 - .46*cosf(2 * M_PI * i/(m_window_width-1.0));

  if (m_coeffs)
    fftw_destroy_plan(m_coeffs);

  m_fftw_datain.resize(m_window_width);
  m_fftw_dataout.resize(m_window_width + 1);
  m_fftw_dataout.back() = 0;
  m_coeffs = fftw_plan_r2r_1d(m_window_width, &m_fftw_datain[0],
                              &m_fftw_dataout[0], FFTW_R2HC, FFTW_ESTIMATE);
}

void
FFTModule::reset_module()
{
  m_first_feature.clear();
  m_last_feature.clear();
  m_last_feature_frame = INT_MIN;
}

void
FFTModule::generate(int frame)
{
  int t;
  
  // NOTE: because of lowpass filtering, (m_window_width PLUS one)
  // samples are fetched from the audio file

  int window_start = (int)(frame * m_window_advance);

  // The first frame is returned for negative frames.
  if (m_copy_borders && frame < 0)
  {
    if (!m_first_feature.empty()) {
      m_buffer[frame].set(m_first_feature);
      return;
    }
    window_start = 0;
  }
  
  // The last whole frame (not containing end of file) is returned for
  // frames after and on the eof.
  else if (m_copy_borders && frame >= m_eof_frame)
  {
    assert(!m_last_feature.empty());
    m_buffer[frame].set(m_last_feature);
    return;
  }

  int window_end = window_start + m_window_width + 1;
  m_reader.fetch(window_start, window_end);

  // EOF during this frame?
  if (m_eof_frame == INT_MAX && m_reader.eof_sample() < window_end) {
    assert(m_reader.eof_sample() >= window_start);
    m_eof_frame = frame;

    if (frame == 0)
      throw std::string("audio shorter than frame");

    if (m_first_feature.empty())
      m_first_feature.resize(m_dim, 0);
    if (m_last_feature.empty())
      m_last_feature.resize(m_dim, 0);

    if (m_copy_borders) {
      assert(!m_last_feature.empty());
      m_buffer[frame].set(m_last_feature);
      return;
    }
  }
  
  // Apply lowpass filtering and hamming window
  for (t = 0; t < m_window_width; t++)
  {
    m_fftw_datain[t] = m_hamming_window[t] * 
      (m_reader[window_start + t + 1] - m_emph_coef * m_reader[window_start + t]);
  }
  
  fftw_execute(m_coeffs);

  // NOTE: fftw returns the imaginary parts in funny order
  FeatureVec target = m_buffer[frame];
  for (t = 0; t < m_window_width / 2; t++)
  {
    target[t] = m_fftw_dataout[t] * m_fftw_dataout[t] + 
      m_fftw_dataout[m_window_width-t] * m_fftw_dataout[m_window_width-t];
    if (m_magnitude)
      target[t] = sqrtf(target[t]);
    if (m_log)
      target[t] = logf(target[t]);
  }
  // The highest frequency component has zero imaginary part
  target[t] = m_fftw_dataout[t] * m_fftw_dataout[t];
  if (m_magnitude)
    target[t] = sqrtf(target[t]);

  if (m_log)
    target[t] = logf(target[t]);
  
  if (m_copy_borders && m_first_feature.empty() && frame <= 0)
    target.get(m_first_feature);

  if (m_copy_borders && frame > m_last_feature_frame) 
  {
    target.get(m_last_feature);
    m_last_feature_frame = frame;
  }
}


//////////////////////////////////////////////////////////////////
// FFTModule
//////////////////////////////////////////////////////////////////

PreModule::PreModule() :
  m_sample_rate(0),
  m_frame_rate(0),
  m_eof_frame(INT_MAX),
  m_legacy_file(0),
  m_file_offset(0),
  m_cur_pre_frame(INT_MAX),
  m_fp(NULL),
  m_last_feature_frame(INT_MIN)
{
  m_type_str = type_str();
}


void
PreModule::set_file(FILE *fp)
{
  int dim;
  m_fp = fp;

  // Read the dimension
  if (m_legacy_file)
  {
    char d;
    if (fread(&d, 1, 1, m_fp) < 1)
      throw std::string("PreModule: Could not read the file.");
    dim  = d;
    m_file_offset = 1;
  }
  else
  {
    if (fread(&dim, sizeof(int), 1, m_fp) < 1)
      throw std::string("PreModule: Could not read the file.");
    m_file_offset = sizeof(int);
  }
  
  // Check that dimension matches that given in configuration
  if (dim != m_dim)
  {
    throw std::string("PreModule: The file has invalid dimension");
  }
  m_eof_frame = INT_MAX; // No EOF frame encountered yet
}


void
PreModule::discard_file(void)
{
  reset_module();
}


bool
PreModule::eof(int frame)
{
  if (frame < m_eof_frame)
    return false;
  return true;

}
int PreModule::last_frame(void)
{
  // FIXME: NOT TESTED!!!
  long cur_pos = ftell(m_fp);
  int last_frame;
  
  if (fseek(m_fp, 0, SEEK_END) < 0)
    throw std::string("PreModule: Could not seek the file.");
  last_frame = (ftell(m_fp)-m_file_offset)/(m_dim*sizeof(float));
  fseek(m_fp, cur_pos, SEEK_SET);
  return last_frame;
}

void
PreModule::get_module_config(ModuleConfig &config)
{
  assert(m_sample_rate > 0);
  config.set("sample_rate", m_sample_rate);
  config.set("frame_rate", m_frame_rate);
  config.set("dim", m_dim);
  if (m_legacy_file)
    config.set("legacy_file", m_legacy_file);
}

void
PreModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;
  
  m_frame_rate = 125;
  m_sample_rate = 16000;
  m_legacy_file = 0;

  config.get("sample_rate", m_sample_rate);
  config.get("frame_rate", m_frame_rate);
  config.get("legacy_file", m_legacy_file);

  if (!config.get("dim", m_dim))
    throw std::string("PreModule: Must set dimension");
}

void
PreModule::reset_module()
{
  m_first_feature.clear();
  m_last_feature.clear();
  m_last_feature_frame = INT_MIN;
  m_cur_pre_frame = INT_MAX;
  m_fp = NULL;
}

void
PreModule::generate(int frame)
{
  int pre_frame = frame;
  FeatureVec target_vec = m_buffer[frame];
  
  if (frame < 0)
  {
    if (!m_first_feature.empty())
    {
      m_buffer[frame].set(m_first_feature);
      return;
    }
    pre_frame = 0;
  }
  else if (frame >= m_eof_frame)
  {
    assert(!m_last_feature.empty());
    m_buffer[frame].set(m_last_feature);
    return;
  }

  if (pre_frame != m_cur_pre_frame + 1)
  {
    // Must seek to the correct place
    if (fseek(m_fp, m_file_offset + pre_frame * m_dim * sizeof(float),
              SEEK_SET) < 0)
      throw std::string("PreModule: Could not seek the file.");
    
  }
  m_cur_pre_frame = pre_frame;

  // Read the frame
  if ((int)fread(&target_vec[0], sizeof(float), m_dim, m_fp) < m_dim)
  {
    if (!feof(m_fp))
      throw std::string("PreModule: Could not read the file");
    // EOF
    m_eof_frame = pre_frame;
    assert(!m_last_feature.empty());
    m_buffer[frame].set(m_last_feature);
    return;
  }

  if (pre_frame > m_last_feature_frame) 
  {
    target_vec.get(m_last_feature);
    m_last_feature_frame = pre_frame;
  }
}


//////////////////////////////////////////////////////////////////
// MelModule
//////////////////////////////////////////////////////////////////

MelModule::MelModule(FeatureGenerator *fea_gen) :
  m_fea_gen(fea_gen)
{
  m_type_str = type_str();
}

void
MelModule::get_module_config(ModuleConfig &config)
{
}

void
MelModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;
  
  m_dim = (int)((21+2)*log10f(1+m_fea_gen->sample_rate()/1400.0) /
                log10f(1+16000/1400.0)-2);
  create_mel_bins();
}


void
MelModule::create_mel_bins(void)
{
  int edges = m_dim + 2;
  float rate = m_fea_gen->sample_rate();
  float mel_step = 2595 * log10f(1.0 + rate / 1400.0) / edges;

  m_bin_edges.resize(edges);
  for (int i = 0; i < edges; i++) {
    m_bin_edges[i] = 1400.0 * (pow(10, (i+1) * mel_step / 2595) - 1)*
      (m_sources.back()->dim()-1) / rate;
  }
}


void
MelModule::generate(int frame)
{
  int t;
  float beg, end, val, scale, sum;
  const FeatureVec data = m_sources.back()->at(frame);
  
  for (int b = 0; b < m_dim; b++)
  {
    val = 0;
    sum = 0;
    beg = m_bin_edges[b] - 1;
    end = m_bin_edges[b+1];
    
    t = (int)std::max(ceilf(beg), 0.0f);
    
    while (t < end)
    {
      scale = (t - beg)/(end - beg);
      val += scale * data[t];
      sum += scale;
      t++;
    }
    beg = end;
    end = m_bin_edges[b+2];
    
    while (t < end)
    {
      scale = (end - t)/(end - beg);
      val += scale * data[t];
      sum += scale;
      t++;
    }
    m_buffer[frame][b] = logf(val/sum + 1);
  }
}


//////////////////////////////////////////////////////////////////
// PowerModule
//////////////////////////////////////////////////////////////////

PowerModule::PowerModule()
{
  m_type_str = type_str();
}

void
PowerModule::get_module_config(ModuleConfig &config)
{
}

void
PowerModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;
  m_dim = 1;
}

void
PowerModule::generate(int frame)
{
  float power = 0;
  int src_dim = m_sources.back()->dim();
  const FeatureVec src = m_sources.back()->at(frame);
  
  for (int i = 0; i < src_dim; i++)
    power += src[i];
  
  m_buffer[frame][0] = log(power + 1e-10);
}


//////////////////////////////////////////////////////////////////
// DCTModule
//////////////////////////////////////////////////////////////////

DCTModule::DCTModule()
{
  m_type_str = type_str();
}

void
DCTModule::get_module_config(ModuleConfig &config)
{
  assert(m_dim > 0);
  config.set("dim", m_dim);
}

void
DCTModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;
  m_dim = 12; // Default dimension

  config.get("dim", m_dim);
  if (m_dim < 1)
    throw std::string("DCTModule: Dimension must be > 0");
}

void
DCTModule::generate(int frame)
{
  const FeatureVec source_fea = m_sources.back()->at(frame);
  FeatureVec target_fea = m_buffer[frame];
  int src_dim = m_sources.back()->dim();
  
  for (int i = 0; i < m_dim; i++)
  {
    target_fea[i] = 0.0;
    for (int b = 0; b < src_dim; b++)
      target_fea[i] += source_fea[b] * cosf((i+1) * (b+0.5) * M_PI / src_dim);
  }
}


//////////////////////////////////////////////////////////////////
// DeltaModule
//////////////////////////////////////////////////////////////////

DeltaModule::DeltaModule()
{
  m_type_str = type_str();
}

void
DeltaModule::get_module_config(ModuleConfig &config)
{
  config.set("width", m_delta_width);
  config.set("normalization", m_delta_norm);
}

void
DeltaModule::set_module_config(const ModuleConfig &config)
{
  m_dim = m_sources.back()->dim();

  m_delta_width = 2; // Default width
  config.get("width", m_delta_width);

  // Set default normalization for deltas.
  // Note! Old delta-features used normalization with (m_delta_width-1)
  m_delta_norm = 2 * m_delta_width*(m_delta_width+1)*(2*m_delta_width+1)/6;
  config.get("normalization", m_delta_norm);

  if (m_delta_width < 1)
    throw std::string("DeltaModule: Delta width must be > 0");

  m_own_offset_left = m_delta_width;
  m_own_offset_right = m_delta_width;
}

void
DeltaModule::generate(int frame)
{
  FeatureVec target_fea = m_buffer[frame];
  int i, k;

  for (i = 0; i < m_dim; i++)
    target_fea[i] = 0;
  
  for (k = 1; k <= m_delta_width; k++)
  {
    const FeatureVec left = m_sources.back()->at(frame-k);
    const FeatureVec right = m_sources.back()->at(frame+k);
    for (i = 0; i < m_dim; i++)
      target_fea[i] += k * (right[i] - left[i]);
  }

  for (i = 0; i < m_dim; i++)
    target_fea[i] /= m_delta_norm;
}


//////////////////////////////////////////////////////////////////
// NormalizationModule
//////////////////////////////////////////////////////////////////

NormalizationModule::NormalizationModule()
{
  m_type_str = type_str();
}

void
NormalizationModule::get_module_config(ModuleConfig &config)
{
  config.set("mean", m_mean);
  config.set("scale", m_scale);
}

void
NormalizationModule::set_module_config(const ModuleConfig &config)
{
  m_dim = m_sources.back()->dim();
  m_own_offset_left = 0;
  m_own_offset_right = 0;

  m_mean.resize(m_dim, 0);
  m_scale.resize(m_dim, 1);

  config.get("mean", m_mean);
  if ((int)m_mean.size() != m_dim)
    throw std::string("NormalizationModule: Invalid mean dimension");

  if (config.exists("var") && config.exists("scale"))
  {
    throw std::string("NormalizationModule: Both scale and var can not be defined simultaneously");
  }
  if (config.get("var", m_scale))
  {
    if ((int)m_scale.size() != m_dim)
      throw std::string("Normalization module: Invalid variance dimension");
    for (int i = 0; i < m_dim; i++)
      m_scale[i] = 1 / sqrtf(m_scale[i]);
  }
  else if (config.get("scale", m_scale))
  {
    if ((int)m_mean.size() != m_dim)
      throw std::string("NormalizationModule: Invalid scale dimension");
  }
}

void
NormalizationModule::set_normalization(const std::vector<float> &mean,
                                       const std::vector<float> &scale)
{
  if ((int)mean.size() != m_dim || (int)scale.size() != m_dim)
    throw std::string("NormalizationModule: The dimension of the new normalization does not match the input dimension");
  for (int i = 0; i < m_dim; i++)
  {
    m_mean[i] = mean[i];
    m_scale[i] = scale[i];
  }
}

void
NormalizationModule::generate(int frame)
{
  const FeatureVec source_fea = m_sources.back()->at(frame);
  FeatureVec target_fea = m_buffer[frame];
  for (int i = 0; i < m_dim; i++)
    target_fea[i] = (source_fea[i] - m_mean[i]) * m_scale[i];
}


//////////////////////////////////////////////////////////////////
// LinTransformModule
//////////////////////////////////////////////////////////////////

LinTransformModule::LinTransformModule()
{
  m_type_str = type_str();
}

void
LinTransformModule::get_module_config(ModuleConfig &config)
{
  assert(m_dim > 0);
  config.set("dim", m_dim);
  // Save the original transformation
  if (m_original_transform.size() > 0)
    config.set("matrix", m_original_transform);
  if (m_original_bias.size() > 0)
    config.set("bias", m_original_bias);
}

void
LinTransformModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;

  m_src_dim = m_sources.back()->dim();
  m_dim = m_src_dim; // Default value

  config.get("matrix", m_transform);
  config.get("bias", m_bias);
  m_original_transform = m_transform;
  m_original_bias = m_bias;
  config.get("dim", m_dim);
  if (m_dim < 1)
    throw std::string("LinTransformModule: Dimension must be > 0");
  
  check_transform_parameters();
}

void
LinTransformModule::set_parameters(const ModuleConfig &config)
{
  m_transform.clear();
  m_bias.clear();
  config.get("matrix", m_transform);
  config.get("bias", m_bias);
  check_transform_parameters();
}

void
LinTransformModule::get_parameters(ModuleConfig &config)
{
  config.set("matrix", m_transform);
  config.set("bias", m_bias);
}

void
LinTransformModule::check_transform_parameters(void)
{
  int r, c, index;
  if (m_transform.size() == 0)
  {
    m_matrix_defined = false;
    // Initialize with identity matrix
    m_transform.resize(m_dim*m_src_dim);
    for (r = index = 0; r < m_dim; r++)
    {
      for (c = 0; c < m_src_dim; c++, index++)
      {
        m_transform[index] = ((r == c) ? 1 : 0);
      }
    }
  }
  else
  {
    m_matrix_defined = true;
    if ((int)m_transform.size() != m_dim*m_src_dim)
      throw std::string("LinTransformModule: Invalid matrix dimension");
  }

  if (m_bias.size() == 0)
  {
    m_bias_defined = false;
    // Initialize with zero vector
    m_bias.resize(m_dim);
    for (c = 0; c < m_dim; c++)
      m_bias[c] = 0;
  }
  else
  {
    m_bias_defined = true;
    if ((int)m_bias.size() != m_dim)
      throw std::string("LinTransformModule: Invalid bias dimension");
  }
}

void
LinTransformModule::generate(int frame)
{
  int index;
  const FeatureVec source_fea = m_sources.back()->at(frame);
  FeatureVec target_fea = m_buffer[frame];

  if (m_matrix_defined)
  {
    for (int i = index = 0; i < m_dim; i++)
    {
      target_fea[i] = 0;
      for (int j = 0; j < m_src_dim; j++, index++)
        target_fea[i] += m_transform[index]*source_fea[j];
    }
  }
  else
  {
    for (int i = 0; i < m_dim; i++)
      target_fea[i] = source_fea[i];
  }
  if (m_bias_defined)
  {
    for (int i = 0; i < m_dim; i++)
      target_fea[i] += m_bias[i];
  }
}


void
LinTransformModule::set_transformation_matrix(std::vector<float> &t)
{
  m_original_transform = t;
  if (t.size() == 0)
  {
    int r, c, index;
    m_transform.resize(m_dim*m_src_dim);
    // Set to identity matrix
    for (r = index = 0; r < m_dim; r++)
      for (c = 0; c < m_src_dim; c++, index++)
        m_transform[index] = ((r == c) ? 1 : 0);
    m_matrix_defined = false;
  }
  else
  {
    if ((int)t.size() != m_dim * m_src_dim)
      throw std::string("LinTransformnModule: The dimension of the new transformation matrix does not match the old dimension");
    for (int i = 0; i < (int)t.size(); i++)
      m_transform[i] = t[i];
    m_matrix_defined = true;
  }
}


void
LinTransformModule::set_transformation_bias(std::vector<float> &b)
{
  m_original_bias = b;
  if (b.size() == 0)
  {
    m_bias_defined = false;
    // Set to zero vector
    m_bias.resize(m_dim);
    for (int i = 0; i < m_dim; i++)
      m_bias[i] = 0;
  }
  else
  {
    if ((int)b.size() != m_dim)
      throw std::string("LinTransformModule: The dimension of the new bias does not match the output dimension");
    for (int i = 0; i < m_dim; i++)
      m_bias[i] = b[i];
    m_bias_defined = true;
  }
}


//////////////////////////////////////////////////////////////////
// MergerModule
//////////////////////////////////////////////////////////////////

MergerModule::MergerModule()
{
  m_type_str = type_str();
}

void
MergerModule::add_source(FeatureModule *source)
{
  // Allow multiple sources
  m_sources.push_back(source);
}

void
MergerModule::get_module_config(ModuleConfig &config)
{
}

void
MergerModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;
  m_dim = 0;
  for (int i = 0; i < (int)m_sources.size(); i++)
    m_dim += m_sources[i]->dim();
}

void
MergerModule::generate(int frame)
{
  FeatureVec target_fea = m_buffer[frame];
  int cur_dim = 0;
  
  for (int i = 0; i < (int)m_sources.size(); i++)
  {
    const FeatureVec source_fea = m_sources[i]->at(frame);
    for (int j = 0; j < source_fea.dim(); j++)
      target_fea[cur_dim++] = source_fea[j];
  }
  assert( cur_dim == m_dim );
}


//////////////////////////////////////////////////////////////////
// MeanSubtractionModule
//////////////////////////////////////////////////////////////////

MeanSubtractorModule::MeanSubtractorModule() :
  m_cur_frame(INT_MAX)
{
  m_type_str = type_str();
}

void
MeanSubtractorModule::get_module_config(ModuleConfig &config)
{
  config.set("left", m_own_offset_left-1);
  config.set("right", m_own_offset_right);
}

void
MeanSubtractorModule::set_module_config(const ModuleConfig &config)
{
  m_dim = m_sources.back()->dim();
  m_cur_mean.resize(m_dim, 0);

  m_own_offset_left = 75; // Default
  config.get("left", m_own_offset_left);

  // We add 1 to m_own_offset_left so that when generating a new frame
  // the furthest context on the left is still available from the
  // previous frame for subtraction from the current mean.
  m_own_offset_left++;

  m_own_offset_right = 75; // Default
  config.get("right", m_own_offset_right);

  if (m_own_offset_left < 1 || m_own_offset_right < 0)
    throw std::string("MeanSubtractorModule: context widths must be >= 0");
  m_width = m_own_offset_left+m_own_offset_right;
}

void
MeanSubtractorModule::reset_module()
{
  m_cur_frame = INT_MAX;
}

void
MeanSubtractorModule::generate(int frame)
{
  FeatureVec target_fea = m_buffer[frame];
  const FeatureVec source_fea = m_sources.back()->at(frame);
  int i, d;

  if (frame == m_cur_frame+1)
  {
    // Update the current mean quickly
    const FeatureVec r = m_sources.back()->at(frame-m_own_offset_left);
    const FeatureVec a = m_sources.back()->at(frame+m_own_offset_right);
    for (d = 0; d < m_dim; d++)
      m_cur_mean[d] += (a[d] - r[d])/m_width;
  }
  else
  {
    // Must go through the entire buffer to determine the mean
    for (i = -m_own_offset_left+1; i<=m_own_offset_right; i++)
    {
      const FeatureVec v = m_sources.back()->at(frame+i);
      for (d = 0; d < m_dim; d++)
        m_cur_mean[d] += v[d];
    }
    for (d = 0; d < m_dim; d++)
      m_cur_mean[d] /= m_width;
  }

  m_cur_frame = frame;

  for (d = 0; d < m_dim; d++)
    target_fea[d] = source_fea[d] - m_cur_mean[d];
}


//////////////////////////////////////////////////////////////////
// ConcatModule
//////////////////////////////////////////////////////////////////

ConcatModule::ConcatModule()
{
  m_type_str = type_str();
}

void
ConcatModule::get_module_config(ModuleConfig &config)
{
  config.set("left", m_own_offset_left);
  config.set("right", m_own_offset_right);
}

void
ConcatModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;

  config.get("left", m_own_offset_left);
  config.get("right", m_own_offset_right);
  
  if (m_own_offset_left < 0 || m_own_offset_right < 0)
    throw std::string("ConcatModule: context spans must be >= 0");

  m_dim = m_sources.back()->dim() * (1+m_own_offset_left+m_own_offset_right);
}

void
ConcatModule::generate(int frame)
{
  FeatureVec target_fea = m_buffer[frame];
  int cur_dim = 0;
  
  for (int i = -m_own_offset_left; i <= m_own_offset_right; i++)
  {
    const FeatureVec source_fea = m_sources.back()->at(frame + i);
    for (int j = 0; j < source_fea.dim(); j++)
      target_fea[cur_dim++] = source_fea[j];
  }
  assert( cur_dim == m_dim );
}


//////////////////////////////////////////////////////////////////
// VtlnModule
//////////////////////////////////////////////////////////////////
VtlnModule::VtlnModule()
{
  m_type_str = type_str();
}

void
VtlnModule::get_module_config(ModuleConfig &config)
{
  if (m_use_pwlin)
  {
    config.set("pwlin_vtln", m_use_pwlin);
    config.set("pwlin_turnpoint", m_pwlin_turn_point);
  }
  if (m_use_slapt)
    config.set("slapt", 1);
  if (m_lanczos_window)
    config.set("lanczos_window", 1);
  config.set("sinc_interpolation_rad", m_sinc_interpolation_rad);
  if (m_all_pass > 0)
    config.set("all-pass", m_all_pass);
}

void
VtlnModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;

  m_dim = m_sources.front()->dim();

  m_use_pwlin = 0;
  m_pwlin_turn_point = 0.8;
  config.get("pwlin_vtln", m_use_pwlin);
  config.get("pwlin_turnpoint", m_pwlin_turn_point);

  m_use_slapt = 0;
  config.get("slapt", m_use_slapt);
  if (m_use_pwlin && m_use_slapt)
    throw std::string("VtlnModule: Can not use both pwlin_vtln and slapt!");

  m_sinc_interpolation_rad = 8;
  config.get("sinc_interpolation_rad", m_sinc_interpolation_rad);

  m_all_pass = 0;
  config.get("all-pass", m_all_pass);
  if (m_use_pwlin && m_all_pass)
    throw std::string("VtlnModule: Can not use both pwlin_vtln and all-pass!");
  
  int lanczos = (m_all_pass?0:1);
  config.get("lanczos_window", lanczos);
  if (lanczos > 0)
    m_lanczos_window = true;
  else
    m_lanczos_window = false;

  if (m_lanczos_window && m_all_pass)
    throw std::string("VtlnModule: Can not use both lanczos_window and all-pass!");
  
  if (!m_use_slapt)
    set_warp_factor(1.0);
  else
  {
    std::vector<float> sparam;
    sparam.push_back(0.0);
    set_slapt_warp(sparam);
  }
}

void
VtlnModule::set_parameters(const ModuleConfig &config)
{
  if (m_use_slapt)
  {
    std::vector<float> sparam;
    sparam.push_back(0.0);
    config.get("slapt_coef", sparam);
    set_slapt_warp(sparam);
  }
  else
  {
    float wf = 1.0;
    config.get("warp_factor", wf);
    set_warp_factor(wf);
  }
}

void
VtlnModule::get_parameters(ModuleConfig &config)
{
  if (m_use_slapt)
    config.set("slapt_coef", m_slapt_params);
  else
    config.set("warp_factor", m_warp_factor);
}

void
VtlnModule::set_warp_factor(float factor)
{
  m_warp_factor = factor;
  if (m_all_pass)
    create_all_pass_blin_transform();
  else if (m_use_pwlin)
    create_pwlin_bins();
  else
    create_blin_bins();
}

void
VtlnModule::set_slapt_warp(std::vector<float> &params)
{
  m_slapt_params = params;
  if (m_all_pass)
    create_all_pass_slapt_transform();
  else
    create_slapt_bins();
}

void
VtlnModule::create_pwlin_bins(void)
{
  int t;
  float border, slope = 0, point = 0;
  bool limit = false;

  border = m_pwlin_turn_point * (float)(m_dim-1);

  m_vtln_bins.resize(m_dim);
  for (t = 0; t < m_dim-1; t++)
  {
    if (!limit)
      m_vtln_bins[t] = m_warp_factor * (float)t;
    else
      m_vtln_bins[t] = slope * (float)t + point;

    if (!limit && (t >= border || m_vtln_bins[t] >= border))
    { 
      slope = ((float)m_dim - 1 - m_vtln_bins[t]) / ((float)m_dim - 1 - t);
      point = (1 - slope) * (float)(m_dim - 1);
      limit = true;
    }
  }
  m_vtln_bins[t] = (float)(m_dim - 1);
  if (m_sinc_interpolation_rad > 0)
    create_sinc_coef_table();
}

void
VtlnModule::create_blin_bins(void)
{
  int t;
  m_vtln_bins.resize(m_dim);
  for (t = 0; t < m_dim-1; t++)
  {
    double nf = M_PI * (double)t / (m_dim - 1);
    m_vtln_bins[t] = t + 2*atan2((m_warp_factor-1)*sin(nf),
                                 1+(1-m_warp_factor)*cos(nf))/M_PI*(m_dim-1);
  }
  m_vtln_bins[t] = m_dim-1;
  if (m_sinc_interpolation_rad > 0)
    create_sinc_coef_table();
}

void
VtlnModule::create_slapt_bins(void)
{
  int t, i;
  m_vtln_bins.resize(m_dim);
  for (t = 0; t < m_dim-1; t++)
  {
    double nf = M_PI * (double)t / (m_dim - 1);
    
    m_vtln_bins[t] = t;
    for (i = 0; i < (int)m_slapt_params.size(); i++)
      m_vtln_bins[t] += m_slapt_params[i]*sin((i+1)*nf)*(m_dim-1);
  }
  m_vtln_bins[t] = m_dim-1;

  if (m_sinc_interpolation_rad > 0)
    create_sinc_coef_table();
}

void
VtlnModule::create_sinc_coef_table(void)
{
  float t;
  m_sinc_coef.resize(m_dim);
  m_sinc_coef_start.resize(m_dim);
  for (int b = 0; b < m_dim; b++)
  {
    int cent = (int)(m_vtln_bins[b]+0.5);
    int min_i = std::max(cent-m_sinc_interpolation_rad, 0);
    int max_i = std::min(cent+m_sinc_interpolation_rad+1, m_dim);
    m_sinc_coef_start[b] = min_i;
    m_sinc_coef[b].clear();
    for (int i = min_i; i < max_i; i++)
    {
      t = util::sinc(i - m_vtln_bins[b]);
      if (m_lanczos_window)
      {
        if (fabs(i-m_vtln_bins[b]) < m_sinc_interpolation_rad)
          t *= util::sinc((i-m_vtln_bins[b])/(float)m_sinc_interpolation_rad);
        else
          t = 0;
      }
      m_sinc_coef[b].push_back(t);
    }
  }
}

void
VtlnModule::create_all_pass_blin_transform(void)
{
  std::vector<double> q1, q, qn;
  Matrix blin_tr(m_dim, m_dim);
  double alpha = m_warp_factor-1;
  double temp;
  int i, j, k;

  q1.resize(m_dim);
  q.resize(m_dim);
  qn.resize(m_dim);
  q1[0] = -alpha;
  temp = 1-alpha*alpha;
  for (i = 1; i < m_dim; i++)
  {
    q1[i] = temp;
    temp *= alpha;
  }
  q[0] = 1;

  blin_tr(0, 0) = 1;
  for (i = 1; i < m_dim; i++)
    blin_tr(i, 0) = 0;

  for (i = 1; i < m_dim; i++)
  {
    for (j = 0; j < m_dim; j++)
    {
      temp = 0;
      for (k = 0; k <= j; k++)
        temp += q[k]*q1[j-k];
      qn[j] = temp;
    }
    q = qn;
    blin_tr(0, i) = 2*q[0];
    for (j = 1; j < m_dim; j++)
      blin_tr(j, i) = q[j];
  }
  set_all_pass_transform(blin_tr);
}

void
VtlnModule::create_all_pass_slapt_transform(void)
{
  std::vector<double> q1, q, qn, f1, cur_f, fn;
  int cur_f_center;
  Matrix slapt_tr(m_dim, m_dim);
  int slapt_order = (int)m_slapt_params.size();
  int i, j, k, len;
  double cur_m = 1;
  double temp;
  int low1, high1, low2, high2;

  f1.resize(2*slapt_order+1);
  for (i = 0; i < slapt_order; i++)
  {
    f1[i] = -m_slapt_params[slapt_order - i - 1]*M_PI/2;
    f1[i+slapt_order+1] = m_slapt_params[i]*M_PI/2;
  }
  f1[slapt_order] = 0;

  q.resize(2*m_dim+1);
  std::fill(q.begin(), q.end(), 0);
  cur_f.push_back(1);
  cur_f_center = 0;

  for (i = 0; i <= 10; i++)
  {
    if (i > 0)
      cur_m = cur_m / (double)i;
    low1 = std::max(0, m_dim-cur_f_center);
    high1 = std::min(2*m_dim+1, m_dim+cur_f_center+1);
    for (j = low1; j < high1; j++)
      q[j] = q[j] + cur_m*cur_f[j - (m_dim + 1) + cur_f_center + 1];
    fn.resize(f1.size()+cur_f.size()-1);
    for (j = 0; j < (int)fn.size(); j++)
    {
      high1 = j;
      if (high1 >= (int)cur_f.size()) {
        high2 = j - cur_f.size() + 1;
        high1 = cur_f.size()-1;
      }
      else {
        high2 = 0;
      }
      low2 = j;
      if (low2 >= (int)f1.size()) {
        low1 = j - f1.size() + 1;
        low2 = f1.size()-1;
      }
      else {
        low1 = 0;
      }
      assert( (high1-low1) == (low2-high2) );
      len = high1 - low1 + 1;
      temp = 0;
      for (k = 0; k < len; k++)
        temp += cur_f[low1+k]*f1[low2-k];
      fn[j] = temp;
    }
    cur_f = fn;
    cur_f_center = (cur_f.size()-1)/2;
  }

  // Make the initial sequence symmetric
  q.pop_back();
  q.pop_back();

  q1 = q;
  slapt_tr(0, 0) = 1;
  for (i = 1; i < m_dim; i++)
    slapt_tr(i, 0) = 0;

  qn.resize(2*m_dim-1);
  for (i = 1; i < m_dim; i++)
  {
    slapt_tr(0, i) = 2*q[m_dim-1];
    for (j = 1; j < m_dim; j++)
      slapt_tr(j, i) = q[m_dim+j-1]+q[m_dim-j-1];

    for (j = m_dim-1; j < 3*m_dim-2; j++)
    {
      high1 = j;
      if (high1 >= (int)q.size()) {
        high2 = j - q.size() + 1;
        high1 = q.size() - 1;
      }
      else {
        high2 = 0;
      }
      low2 = j;
      if (low2 >= (int)q1.size()) {
        low1 = j - q1.size() + 1;
        low2 = q1.size()-1;
      }
      else {
        low1 = 0;
      }
      assert( (high1-low1) == (low2-high2) );
      len = high1 - low1 + 1;
      temp = 0;
      for (k = 0; k < len; k++)
        temp += q[low1+k]*q1[low2-k];
      qn[j - m_dim + 1] = temp;
    }
    q = qn;
  }

  set_all_pass_transform(slapt_tr);
}

void
VtlnModule::set_all_pass_transform(Matrix &trmat)
{
  Matrix dct(m_dim, m_dim);
  Matrix final(m_dim, m_dim);
  Matrix temp_m(m_dim, m_dim);
  int i, j;
  
  // Make DCT matrix
  for (i = 0; i < m_dim; i++)
  {
    for (j = 0; j < m_dim; j++)
       dct(i, j) = cos(i*(j+0.5)*M_PI/m_dim);
  }
  Blas_Mat_Mat_Mult(trmat, dct, temp_m, 1.0, 0.0);

  // Make inverse DCT matrix
  for (i = 0; i < m_dim; i++)
  {
    dct(i, 0) = 1.0/m_dim;
    for (j = 1; j < m_dim; j++)
       dct(i, j) = cos((i+0.5)*j*M_PI/m_dim)*2/m_dim;
  }
  Blas_Mat_Mat_Mult(dct, temp_m, final, 1.0, 0.0);

  // Fill the interpolation matrix
  m_sinc_coef.resize(m_dim);
  m_sinc_coef_start.resize(m_dim);
  for (i = 0; i < m_dim; i++)
  {
    m_sinc_coef_start[i] = 0;
    m_sinc_coef[i].clear();
    for (j = 0; j < m_dim; j++)
      m_sinc_coef[i].push_back(final(i,j));
  }
}

void
VtlnModule::generate(int frame)
{
  float p;
  const FeatureVec data = m_sources.back()->at(frame);
  FeatureVec target = m_buffer[frame];

  if (m_sinc_interpolation_rad > 0)
  {
    for (int b = 0; b < m_dim; b++)
    {
      double t = 0;
      int i, di;
      for (i = 0, di = m_sinc_coef_start[b];
           i < (int)m_sinc_coef[b].size(); i++, di++)
          t += data[di]*m_sinc_coef[b][i];
      target[b] = std::max((float)t, 0.0f);
    }
  }
  else
  {
    for (int b = 0; b < m_dim; b++)
    {
      p = ceil(m_vtln_bins[b]) - m_vtln_bins[b]; 
      
      target[b] = p*data[(int)floor(m_vtln_bins[b])] + 
        (1-p)*data[(int)ceil(m_vtln_bins[b])];
    }
  }
}


//////////////////////////////////////////////////////////////////
// SRNormModule
//////////////////////////////////////////////////////////////////

SRNormModule::SRNormModule()
{
  m_type_str = type_str();
}

void
SRNormModule::get_module_config(ModuleConfig &config)
{
  config.set("in_frames", m_in_frames);
  config.set("out_frames", m_out_frames);
  config.set("lanczos_order", m_lanczos_order);
}

void
SRNormModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;

  m_in_frames = 0;
  m_out_frames = 0;
  config.get("in_frames", m_in_frames);
  config.get("out_frames", m_out_frames);

  if (m_in_frames == 0 || m_out_frames == 0)
    throw std::string("SRNormModule: Must set both in_frames and out_frames.");
  
  m_frame_dim = m_sources.front()->dim() / m_in_frames;
  if (m_sources.front()->dim() % m_in_frames != 0)
    throw std::string("SRNormModule: in_frames does not match with the input dimension");

  if (m_in_frames%2 == 0)
    fprintf(stderr, "SRNormModule: Warning: Number of input frames is even");
  if (m_in_frames%2 == 0)
    fprintf(stderr, "SRNormModule: Warning: Number of output frames is even");
  
  m_dim = m_out_frames * m_frame_dim;

  m_lanczos_order = 4;
  config.get("lanczos_order", m_lanczos_order);
  if (m_lanczos_order < 1)
    throw std::string("SRNormModule: lanczos_order must be positive.");

  // Debug options?
  float sr = 1.0;
  config.get("speech_rate", sr);
  set_speech_rate(sr);
}

void
SRNormModule::set_parameters(const ModuleConfig &config)
{
  float sr = 1.0;
  config.get("speech_rate", sr);
  set_speech_rate(sr);
}

void
SRNormModule::get_parameters(ModuleConfig &config)
{
  config.set("speech_rate", m_speech_rate);
}

void
SRNormModule::set_speech_rate(float sr)
{
  float in_cent = (float)(m_in_frames-1)/2;
  float out_cent = (float)(m_out_frames-1)/2;
  float target_pos;

  m_speech_rate = sr; // Fast: >1, slow: <1

  m_coef.resize(m_out_frames);
  m_interpolation_start.resize(m_out_frames);
  for (int i = 0; i < m_out_frames; i++)
  {
    target_pos = (i - out_cent)/m_speech_rate + in_cent;

    int cent = (int)roundf(target_pos);
    int interp_start = std::max(cent-m_lanczos_order, 0);
    int interp_end = std::min(cent+m_lanczos_order+1, m_in_frames);
    m_interpolation_start[i] = interp_start;
    m_coef[i].clear();
    
    for (int j = interp_start; j < interp_end; j++)
    {
      float t = util::sinc(j - target_pos);
      if (fabs(j - target_pos) < m_lanczos_order)
        t *= util::sinc((j-target_pos)/(float)m_lanczos_order);
      else
        t = 0;
      m_coef[i].push_back(t);
    }
  }
}


void
SRNormModule::generate(int frame)
{
  const FeatureVec data = m_sources.back()->at(frame);
  FeatureVec target = m_buffer[frame];
  double t;
  int i, j, d, fi;

  for (i = 0; i < m_out_frames; i++)
  {
    for (d = 0; d < m_frame_dim; d++)
    {
      t = 0;
      for (j = 0, fi = m_interpolation_start[i];
           j < (int)m_coef[i].size(); j++, fi++)
        t += m_coef[i][j]*data[fi*m_frame_dim + d];

      // Assuming positive input/output
      target[i*m_frame_dim + d] = std::max((float)t, 0.0f);
    }
  }
}

