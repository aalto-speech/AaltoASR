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

PhnReader::PhnReader()
  : m_file(NULL), m_speaker_phns(false), m_state_num_labels(false)
{
}

void
PhnReader::open(std::string filename) 
{
  m_current_line = 0;
  m_first_line = 0;
  m_last_line = 0;
  m_first_sample = 0;
  m_last_sample = 0;

  if (m_file)
    fclose(m_file);

  m_file = fopen(filename.c_str(), "r");
  if (!m_file) {
    fprintf(stderr, "PhnReader::open(): could not open %s\n", 
	    filename.c_str());
    perror("error");
    exit(1);
  }
}

void
PhnReader::reset_file(void)
{
  assert( m_file != NULL );
  fseek(m_file, 0, SEEK_SET);

  m_current_line = 0;
  if (m_first_sample > 0)
    set_sample_limit(m_first_sample, m_last_sample);
  if (m_first_line > 0)
  {
    int fs;
    set_line_limit(m_first_line, m_last_line, &fs);
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
PhnReader::set_line_limit(int first_line, int last_line, 
			  int *first_sample) 
{
  Phn phn;
  if (!m_state_num_labels)
  {
    m_first_line = first_line;
    m_last_line = last_line;

    while (m_current_line < m_first_line) 
      next(phn);
  }
  *first_sample = phn.start;
}

void
PhnReader::set_sample_limit(int first_sample, int last_sample) 
{
  Phn phn;
  if (!m_state_num_labels)
  {
    m_first_sample = first_sample;
    m_last_sample = last_sample;
    long oldpos = ftell(m_file);
    long curpos = oldpos;
    
    while (next(phn)) {
      oldpos = curpos;
      curpos = ftell(m_file);
      if (phn.end < 0 || phn.end > m_first_sample) {
        fseek(m_file, oldpos, SEEK_SET);
        return;
      }
    }
  }
}

void
PhnReader::set_speaker_phns(bool sphn)
{
  m_speaker_phns = sphn;
}

bool
PhnReader::next(Phn &phn)
{  

  // Read line at time
  if (!str::read_line(&m_line, m_file)) {
    if (ferror(m_file)) {
      fprintf(stderr, "PhnReader::next(): read error on line %d: %s\n",
	      m_current_line, strerror(errno));
      exit(1);
    }

    if (feof(m_file))
      return false;
    
    assert(false);
  }

  if (!m_state_num_labels && m_last_line > 0 && m_current_line > m_last_line)
    return false; 

  // Parse the line in fields.
  str::chomp(&m_line);
  std::vector<std::string> fields;

  // If speakered phns are used, there is an additional
  // field containing the speaker ID.
  int ID_field = 0;
  if(m_speaker_phns)
    ID_field = 1;

  // If the first char is digit, we have start and end fields.
  if (isdigit(m_line[0])) {
    str::split(&m_line, " \t", true, &fields, 4 + ID_field);
    bool ok = true;

    if (fields.size() > 2) {

      // read start & end

      phn.start = str::str2long(&fields[0], &ok);
      phn.end = str::str2long(&fields[1], &ok);

      // read state

      phn.state = -1; // state default value !

      if (strchr(fields[2].c_str(), '.') != NULL){
	phn.state = atoi( (const char*)(strchr(fields[2].c_str(), '.') + 1) );
	fields[2].erase(fields[2].find('.', 0), 2);
      }

    }
    else
      ok = false;

    if (!ok) {
      fprintf(stderr, 
	      "PhnReader::next(): invalid start or end time on line %d:\n"
	      "%s\n", m_current_line, m_line.c_str());
      exit(1);
    }

    fields.erase(fields.begin(), fields.begin() + 2);
  }

  // Otherwise we have just label and comments. (and possibly speaker ID)
  else {
    str::split(&m_line, " \t", true, &fields, 2 + ID_field);
    phn.start = -1;
    phn.end = -1;
  }

  // Is the current starting time out of requested range?
  if (!m_state_num_labels && m_last_sample > 0 && phn.start > m_last_sample)
    return false;

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

  if ((int)fields.size() > 1 + ID_field)
    phn.comment = fields[1 + ID_field];
  else
    phn.comment = "";
  
  // Read speaker ID

  if (m_speaker_phns)
    phn.speaker = fields[1]; 
  else
    phn.speaker = "";

  m_current_line++;
  return true;
}

