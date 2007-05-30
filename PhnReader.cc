#include <ctype.h>
#include <vector>
#include <string>
#include <errno.h>
#include <string.h>
#include <sstream>
#include <cstdlib>
#include <assert.h>

#include "PhnReader.hh"
#include "str.hh"


PhnReader::Phn::Phn()
  : start(0), end(0)
{
}

PhnReader::PhnReader(HmmSet *model)
  : m_file(NULL), m_model(model),
    m_state_num_labels(false), m_relative_sample_numbers(false)
{
  Segmentator::IndexProbPair p(-1, 1);
  set_frame_rate(125); // Default frame rate

  // Initialize the current state and its probability
  m_cur_pdf.push_back(p);
}

PhnReader::~PhnReader()
{
  close();
}

void
PhnReader::open(std::string filename) 
{
  m_current_line = 0;
  m_first_line = 0;
  m_last_line = 0;
  m_first_frame = 0;
  m_last_frame = 0;
  m_current_frame = -1;
  m_eof_flag = false;

  close();

  m_file = fopen(filename.c_str(), "r");
  if (!m_file) {
    fprintf(stderr, "PhnReader::open(): could not open %s\n", 
	    filename.c_str());
    perror("error");
    exit(1);
  }
}

void
PhnReader::close()
{
  if (m_file)
    fclose(m_file);
  m_file = NULL;
}

void
PhnReader::reset(void)
{
  assert( m_file != NULL );
  fseek(m_file, 0, SEEK_SET);

  m_current_line = 0;
  m_current_frame = -1;
  m_eof_flag = false;
  if (m_first_frame > 0)
    set_frame_limits(m_first_frame, m_last_frame);
  if (m_first_line > 0)
    set_line_limits(m_first_line, m_last_line, NULL);
}


void
PhnReader::set_line_limits(int first_line, int last_line, 
                           int *first_frame) 
{
  Phn phn;
  m_first_line = first_line;
  m_last_line = last_line;

  while (m_current_line < m_first_line) 
    next_phn_line(phn);
  
  if (first_frame != NULL)
  {
    *first_frame = phn.start;
    if (m_relative_sample_numbers)
      (*first_frame) += m_first_frame;
  }
}

void
PhnReader::set_frame_limits(int first_frame, int last_frame) 
{
  Phn phn;
  
  m_first_frame = first_frame;
  m_last_frame = last_frame;
  if (!m_relative_sample_numbers)
  {
    long oldpos = ftell(m_file);
    long curpos = oldpos;
    
    while (next_phn_line(phn)) {
      oldpos = curpos;
      curpos = ftell(m_file);
      if (phn.end < 0 || phn.end > m_first_frame)
      {
        fseek(m_file, oldpos, SEEK_SET);
        m_current_line--;
        return;
      }
    }
  }
}


bool
PhnReader::init_utterance_segmentation(void)
{
  m_eof_flag = false;
  if (!next_phn_line(m_cur_phn))
    m_eof_flag = true;
  return !m_eof_flag;
}


bool
PhnReader::next_frame(void)
{
  int cur_state_index = -1;

  assert( m_model != NULL );
  
  if (m_eof_flag)
    return false;

  // Segmentator object must reset the model cache during next_frame()
  m_model->reset_cache();
  
  if (m_current_frame == -1)
  {
    // Initialize the current frame to the beginning of the file
    m_current_frame = m_cur_phn.start;
    assert( m_cur_phn.end >= m_current_frame );
  }
  else
  {
    m_current_frame++;
  }

  assert( m_current_frame >= m_cur_phn.start );
  assert( m_current_frame <= m_cur_phn.end );

  if (m_state_num_labels)
  {
    cur_state_index = m_cur_phn.state;
  }
  else
  {
    if (m_cur_phn.state < 0)
      throw std::string("PhnReader::next_frame(): A state segmented phn file is required");
    Hmm &hmm = m_model->hmm(m_model->hmm_index(m_cur_phn.label[0]));
    cur_state_index = hmm.state(m_cur_phn.state);
  }
  m_cur_pdf.back().index = m_model->emission_pdf_index(cur_state_index);

  bool new_phn_loaded = false;
  Phn prev_phn = m_cur_phn;

  // Do we need to load more phn lines?
  while (m_current_frame+1 >= m_cur_phn.end)
  {
    if (!next_phn_line(m_cur_phn))
    {
      m_eof_flag = true; // For the next call
      break;
    }
    new_phn_loaded = true;
  }

  if (m_collect_transitions)
  {
    m_transition_info.clear();
    if (!m_eof_flag) // Not the last frame
    {
      std::vector<int> &tr_index=m_model->state(cur_state_index).transitions();
      int transition_index = -1;
      
      if (new_phn_loaded)
      {
        // Out transition
        if (m_state_num_labels)
        {
          // We don't have information which transition it is, select the first
          // out transition
          for (int i = 0; i < (int)tr_index.size(); i++)
            if (m_model->transition(tr_index[i]).target_offset != 0)
            {
              transition_index = tr_index[i];
              break;
            }
        }
        else
        {
          int cur_state = prev_phn.state;
          Hmm &cur_hmm = m_model->hmm(m_model->hmm_index(prev_phn.label[0]));

          // Find the correct transition
          for (int i = 0; i < (int)tr_index.size(); i++)
          {
            int next_state =
              m_model->transition(tr_index[i]).target_offset+cur_state;

            if ((next_state >= cur_hmm.num_states() &&
                 m_cur_phn.state == 0) ||
                (next_state == m_cur_phn.state))
            {
              transition_index = tr_index[i];
              break;
            }
          }
        }
      }
      else
      {
        // Self transition
        for (int i = 0; i < (int)tr_index.size(); i++)
          if (m_model->transition(tr_index[i]).target_offset == 0)
          {
            transition_index = tr_index[i];
            break;
          }
      }
      if (transition_index == -1)
      {
        throw std::string("PhnReader::next_frame(): Correct transition was not found");
      }
      if (transition_index != -1)
      {
        Segmentator::IndexProbPair new_transition(transition_index, 1);
        m_transition_info.push_back(new_transition);
      }
    }
  }

  return true;
}


bool
PhnReader::next_phn_line(Phn &phn)
{  
  // Read line at time
  if (!str::read_line(&m_line, m_file)) {
    if (ferror(m_file)) {
      throw str::fmt(1024,
                     "PhnReader::next_phn_line(): read error on line %d: %s\n",
                     m_current_line, strerror(errno));
    }

    if (feof(m_file))
      return false;
    
    assert(false);
  }

  if (m_last_line > 0 && m_current_line >= m_last_line)
    return false; 

  // Parse the line in fields.
  str::chomp(&m_line);
  std::vector<std::string> fields;

  // If the first char is digit, we have start and end fields.
  if (isdigit(m_line[0])) {
    str::split(&m_line, " \t", true, &fields, 4);
    bool ok = true;

    if (fields.size() > 2) {

      // read start & end

      phn.start = (int)(str::str2long(&fields[0], &ok)/m_samples_per_frame);
      phn.end = (int)(str::str2long(&fields[1], &ok)/m_samples_per_frame);

      // read state

      phn.state = -1; // state default value !

      if (strchr(fields[2].c_str(), '.') != NULL){
	phn.state = atoi( (const char*)(strchr(fields[2].c_str(), '.') + 1) );
	fields[2].erase(fields[2].find('.', 0), 2);
      }

    }
    else
      ok = false;

    if (!ok || phn.start > phn.end) {
      throw str::fmt(
        1024,
        "PhnReader::next_phn_line(): invalid start or end time on line %d:\n"
        "%s\n", m_current_line, m_line.c_str());
    }

    fields.erase(fields.begin(), fields.begin() + 2);
  }

  // Otherwise we have just label and comments.
  else {
    str::split(&m_line, " \t", true, &fields, 2);
    phn.start = -1;
    phn.end = -1;
  }

  if (m_relative_sample_numbers && phn.start >= 0)
  {
    phn.start += m_first_frame;
    phn.end += m_first_frame;
  }
  
  // Is the current starting time out of requested range?
  if (m_last_frame > 0)
  {
    if (phn.start >= m_last_frame)
      return false;
    if (phn.end >= m_last_frame)
      phn.end = m_last_frame;
  }

  if (m_first_frame > 0 && phn.start < m_first_frame && phn.start >= 0)
  {
    phn.start = m_first_frame;
    assert( phn.start > phn.end );
  }

  // Read label and comments
  phn.label.clear();
  if (m_state_num_labels)
  {
    phn.state = atoi(fields[0].c_str()); // State number instead of label
  }
  else
  {
    str::split(&fields[0], ",", false, &phn.label);
  }

  if ((int)fields.size() > 1)
    phn.comment = fields[1];
  else
    phn.comment = "";
  
  m_current_line++;
  return true;
}

