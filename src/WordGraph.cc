#include "WordGraph.hh"

#define LINE_SIZE 256

WordGraph::WordGraph(FILE *file, Vocabulary *vocab) {
  m_file = file;
  m_vocab = vocab;
  if (m_file == NULL || m_vocab == NULL)
    ok = false;
  else ok = true;
  m_read_mode = NODES;
  m_node_table = NULL;
  m_frame_table = NULL;
  m_node_count = 0;
  m_edge_count = 0;
  m_frame_count = 0;
}

WordGraph::~WordGraph() {

  for (int i = 0; i < m_edge_count; i++)
    delete m_edge_vector[i];    
  
  if (m_node_table != NULL) {
    for (int j = 0; j < m_node_count; j++)
      delete m_node_table[j];
    delete m_node_table;
  }

  for (int k = 0; k < m_frame_count; k++)
    delete m_frame_table[k];
  delete m_frame_table;
}

int WordGraph::read() {

  int index;
  char line[LINE_SIZE];
  char word[LINE_SIZE];
  WordGraph::Node *temp_node = NULL;
  WordGraph::Edge *temp_edge = NULL;

  while(ok && !feof(m_file) && fgets(line, LINE_SIZE, m_file) != NULL) {

    // discard comment & empty lines
    if (line[0] == '#' || line[0] == '\n' || line[0] == ' ');

    // node/edge parsing
    else if (isdigit(line[0])) {
      
      // node parsing
      if (m_read_mode == NODES) {

	temp_node = new WordGraph::Node;
	
	if (sscanf(line, "%i%*[ ]%[^ ]%*[ ]%u%*u%*u", 
		   &index, word, &temp_node->frame) == 3) 
	  {
	    temp_node->word_id = m_vocab->index(word);
	    m_node_table[index] = temp_node;
	    m_frame_table[temp_node->frame]->push_back(*temp_node);
	  }
	else {
	  delete temp_node;
	  ok = false;		  
	}
      }
      
      // edge parsing
      else if (m_read_mode == EDGES) {

	temp_edge = new WordGraph::Edge;
	
	if (sscanf(line, "%u%*[ ]%u%*[ ]%f",
		   &index, &temp_edge->target_node, &temp_edge->ac_log_prob) == 3) 
	  {
	    m_node_table[index]->edges.push_back(m_edge_count);
	    m_edge_vector.push_back(temp_edge);
	    m_edge_count++;
	  }
	else {
	  delete temp_edge;
	  ok = false;	
	}
      }      
    }

    // command parsing
    else if (isalpha(line[0])) {

      // 'Frames <frame_count>' keyword
      if (strncmp(line, "Frames", 6) == 0) {
	sscanf(line, "%*[^ ]%*[ ]%u", &m_frame_count);
	m_frame_count++;
	m_frame_table = new std::vector<WordGraph::Node>*[m_frame_count];
	for (int i = 0; i < m_frame_count; i++)
	  m_frame_table[i] = new std::vector<WordGraph::Node>;
      }

      // 'Nodes <node_count>' keyword
      else if (strncmp(line, "Nodes", 5) == 0) {
	sscanf(line, "%*[^ ]%*[ ]%u", &m_node_count);
	m_read_mode = NODES;
	m_node_table = new WordGraph::Node*[m_node_count];
      }

      // 'Initial <init_count>' keyword
      else if (strncmp(line, "Initial", 5) == 0) {
      }

      // 'Final <final_count>' keyword
      else if (strncmp(line, "Final", 5) == 0) {
      }

      // 'BestSegAscr <?_number>' keyword
      else if (strncmp(line, "BestSegAscr", 5) == 0) {
	sscanf(line, "%*[^ ] %u", &m_best_seg_ascr);
      }

      // 'Edges' keyword
      else if (strncmp(line, "Edges", 5) == 0) {
	m_read_mode = EDGES;
      }
    }
    
    // error
    else {
      ok = false;
    }
  } 
  
  if (m_file != NULL)
    if (!fclose(m_file)) ok = false;
  
  return ok;
}

WordGraph::Node & WordGraph::node(int index) {
  if (index < 0 || index >= m_node_count)
    throw RangeError();
  return *m_node_table[index];
}

WordGraph::Edge & WordGraph::edge(int index) {
  if (index < 0 || index >= m_edge_count)
    throw RangeError();
  return *m_edge_vector[index];
}

std::vector<WordGraph::Node> & WordGraph::frame(int frame) {
  if (frame < 0 || frame >= m_frame_count)
    throw RangeError();
  return *m_frame_table[frame];
}

int WordGraph::node_count() {
  return m_node_count;
}

int WordGraph::edge_count() {
  return m_edge_count;
}

int WordGraph::frame_count() {
  return m_frame_count;
}
