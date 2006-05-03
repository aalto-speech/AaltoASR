#include "FeatureModules.hh"
#include "FeatureGenerator.hh"

#include <math.h>


FeatureModule::FeatureModule() :
  m_own_offset_left(-1),
  m_own_offset_right(-1),
  m_req_offset_left(0),
  m_req_offset_right(0),
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
        m_sources[i]->set_buffer(m_req_offset_left + m_own_offset_left,
                                 m_req_offset_right + m_own_offset_right);
    }
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
FeatureModule::link(FeatureModule *source)
{
  if (m_sources.size() > 0)
  {
    throw std::string("Multiple links are not allowed for this module");
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


//////////////////////////////////////////////////////////////////
// FFTModule
//////////////////////////////////////////////////////////////////

FFTModule::FFTModule(FeatureGenerator *fea_gen) :
  m_fea_gen(fea_gen),
  m_sample_rate(0),
  m_frame_rate(0),
  m_eof_frame(INT_MAX),
  m_window_advance(0),
  m_window_width(0),
  m_coeffs(NULL)
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
  else if (m_fea_gen->audio_format() == FeatureGenerator::AF_WAV)
  {
    m_reader.open(fp);
  }
  else
  {
    throw std::string("Trying to open a non-wave file");
  }
  // Check that sample rate matches that given in configuration
  if (m_reader.sample_rate() != m_sample_rate)
  {
    throw std::string(
      "File sample rate does not match the model configuration");
  }
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

void
FFTModule::get_module_config(ModuleConfig &config)
{
  assert(m_sample_rate > 0);
  config.set("sample_rate", m_sample_rate);
}

void
FFTModule::set_module_config(const ModuleConfig &config)
{
  m_own_offset_left = 0;
  m_own_offset_right = 0;
  
  m_frame_rate = 125;

  if (!config.get("sample_rate", m_sample_rate))
    throw std::string("FFTModule: Must set sample rate");

  m_window_width = (int)(m_sample_rate/62.5);
  m_window_advance = (int)(m_sample_rate/125);
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
FFTModule::generate(int frame)
{
  // Fetch m_window_width samples (+1 for lowpass filtering)
  m_reader.fetch(frame * m_window_advance,
                 frame * m_window_advance + m_window_width + 1);

  if (m_reader.eof_sample() < INT_MAX)
  {
    m_eof_frame = std::max((m_reader.eof_sample() - m_window_width - 1) /
                           m_window_advance, 0);
  }
  
  // Apply lowpass filtering and hamming window
  int win_start = frame * m_window_advance;
  for (int t = 0; t < m_window_width; t++)
  {
    m_fftw_datain[t] = m_hamming_window[t] * 
      (m_reader[win_start + t + 1] - 0.95 * m_reader[win_start + t]);
  }
  
  fftw_execute(m_coeffs);

  // NOTE: fftw returns the imaginary parts in funny order
  FeatureVec target = m_buffer[frame];
  for (int t = 0; t <= m_window_width / 2; t++)
  {
    target[t] = m_fftw_dataout[t] * m_fftw_dataout[t] + 
      m_fftw_dataout[m_window_width-t] * m_fftw_dataout[m_window_width-t];
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
  m_own_offset_left = m_delta_width;
  m_own_offset_right = m_delta_width;
  m_dim = m_sources.back()->dim();
  m_delta_width = 2; // Default width
  // Set default normalization for deltas.
  // Note! Old delta-features used normalization with (m_delta_width-1)
  m_delta_norm = 2 * m_delta_width*(m_delta_width+1)*(2*m_delta_width+1)/6;

  config.get("width", m_delta_width);
  config.get("normalization", m_delta_norm);

  if (m_delta_width < 1)
    throw std::string("DeltaModule: Delta width must be > 0");
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
NormalizationModule::generate(int frame)
{
  const FeatureVec source_fea = m_sources.back()->at(frame);
  FeatureVec target_fea = m_buffer[frame];
  for (int i = 0; i < m_dim; i++)
    target_fea[i] = (source_fea[i] - m_mean[i]) * m_scale[i];
}


//////////////////////////////////////////////////////////////////
// TransformationModule
//////////////////////////////////////////////////////////////////

TransformationModule::TransformationModule()
{
  m_type_str = type_str();
}

void
TransformationModule::get_module_config(ModuleConfig &config)
{
  assert(m_dim > 0);
  config.set("matrix", m_transform);
  config.set("dim", m_dim);
}

void
TransformationModule::set_module_config(const ModuleConfig &config)
{
  int r, c, index;
  
  m_own_offset_left = 0;
  m_own_offset_right = 0;

  m_src_dim = m_sources.back()->dim();
  m_dim = m_src_dim; // Default value

  config.get("matrix", m_transform);
  config.get("dim", m_dim);
  if (m_dim < 1)
    throw std::string("TransformationModule: Dimension must be > 0");
  
  if (m_transform.size() == 0)
  {
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
    if (m_dim == 0)
      throw std::string("TransformationModule: Must set the output dimension");
    else if ((int)m_transform.size() != m_dim*m_src_dim)
      throw std::string("TransformationModule: Invalid matrix dimension");
  }
}


void
TransformationModule::generate(int frame)
{
  int index;
  const FeatureVec source_fea = m_sources.back()->at(frame);
  FeatureVec target_fea = m_buffer[frame];
  for (int i = index = 0; i < m_dim; i++)
  {
    target_fea[i] = 0;
    for (int j = 0; j < m_src_dim; j++, index++)
      target_fea[i] += m_transform[index]*source_fea[j];
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
MergerModule::link(FeatureModule *source)
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

