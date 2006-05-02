#include "FeatureGenerator.hh"



FeatureGenerator::FeatureGenerator(void) :
  m_base_module(NULL),
  m_last_module(NULL),
  m_eof_on_last_frame(false)
{
}

FeatureGenerator::~FeatureGenerator()
{
  for (int i = 0; i < (int)m_modules.size(); i++)
    delete m_modules[i];
}

void
FeatureGenerator::open(const std::string &filename, int raw_sample_rate)
{
  if (raw_sample_rate > 0)
  {
    m_audio_format = AF_RAW;
  }
  else if (filename.find(".htk") != std::string::npos)
  {
    m_audio_format = AF_HTK;
  } 
  else if (filename.find(".pre") != std::string::npos)
  {
    m_audio_format = AF_PRE;
  }
  else
  {
    m_audio_format = AF_WAV;
  }

  if ((m_file = fopen(filename.c_str(), "rb")) == NULL)
  {
    throw std::string("Could not open file ")+filename;
  }
  
  assert( m_base_module != NULL );
  m_base_module->set_file(m_file);
}


void
FeatureGenerator::close(void)
{
  if (m_file != NULL)
  {
    m_base_module->discard_file();
  }
}


void
FeatureGenerator::load_configuration(FILE *file)
{
  m_base_module = new FFTModule(this);
  m_modules.push_back(m_base_module);
  m_modules.push_back(new MelModule(this));
  m_modules.push_back(new PowerModule);
  m_modules.push_back(new DCTModule);
  m_modules.push_back(new MergerModule);
  m_modules.push_back(new DeltaModule);
  m_modules.push_back(new DeltaModule);
  m_last_module = new MergerModule;
  m_modules.push_back(m_last_module);

  m_modules[1]->link(m_modules[0]); // MEL->FFT
  m_modules[2]->link(m_modules[0]); // Power->FFT
  m_modules[3]->link(m_modules[1]); // DCT->MEL
  m_modules[4]->link(m_modules[3]); // Merger1->DCT
  m_modules[4]->link(m_modules[2]); // Merger1->Power
  m_modules[5]->link(m_modules[4]); // Delta1->Merger1
  m_modules[6]->link(m_modules[5]); // Delta2->Delta1
  m_modules[7]->link(m_modules[4]); // Merger2->Merger1
  m_modules[7]->link(m_modules[5]); // Merger2->Delta1
  m_modules[7]->link(m_modules[6]); // Merger2->Delta2


  std::vector<struct ConfigPair> empty;
  for (int i=0; i < 8; i++)
    m_modules[i]->configure(empty);
  m_last_module->set_buffer(0, 0);
}


void
FeatureGenerator::write_configuration(FILE *file)
{
}

