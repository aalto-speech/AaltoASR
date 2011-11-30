#include <cstddef>  // NULL
#include "WordGraph.hh"

#define LINE_SIZE 256

WordGraph::WordGraph(FILE *file, Vocabulary *vocab) {
  m_file = file;
  m_vocab = vocab;
  if (m_file == NULL || m_vocab == NULL)
    ok = false;
  else ok = true;
  m_read_mode = NODES;
  m_node_count = 0;
  m_edge_count = 0;
  m_frame_count = 0;
  m_best_seg_ascr = 0;
}

WordGraph::~WordGraph() {
  m_nodes.clear();
  m_edges.clear();
  m_frames.clear();
}

int WordGraph::read() {

  int index;
  char line[LINE_SIZE];
  char word[LINE_SIZE];

  while(ok && !feof(m_file) && fgets(line, LINE_SIZE, m_file) != NULL) {

    // discard comment & empty lines
    if (line[0] == '#' || line[0] == '\n' || line[0] == ' ');

    // node/edge parsing
    else if (isdigit(line[0])) {
      
      // node parsing
      if (m_read_mode == NODES) {

	WordGraph::Node new_node;
	
	if (sscanf(line, "%i%*[ ]%[^ ]%*[ ]%u%*u%*u", 
		   &index, word, &new_node.frame) == 3) 
	  {
	    new_node.word_id = m_vocab->index(word);
	    m_nodes[index] = new_node;
	    m_frames[new_node.frame].push_back(new_node);
	  }
	else {
	  ok = false;		  
	}
      }
      
      // edge parsing
      else if (m_read_mode == EDGES) {

	WordGraph::Edge new_edge;
	
	if (sscanf(line, "%u%*[ ]%u%*[ ]%f",
		   &index, &new_edge.target_node, &new_edge.ac_log_prob) == 3) 
	  {
	    m_nodes[index].edges.push_back(m_edge_count);
	    m_edges.push_back(new_edge);
	    m_edge_count++;
	  }
	else {
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
	m_frames.reserve(m_frame_count);
	for (int i = 0; i < m_frame_count; i++) {
	  std::vector<WordGraph::Node> new_frame;
	  m_frames[i] = new_frame;	  
	}	  
      }

      // 'Nodes <node_count>' keyword
      else if (strncmp(line, "Nodes", 5) == 0) {
	sscanf(line, "%*[^ ]%*[ ]%u", &m_node_count);
	m_read_mode = NODES;
	m_nodes.reserve(m_node_count);
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
  return m_nodes[index];
}

WordGraph::Edge & WordGraph::edge(int index) {
  if (index < 0 || index >= m_edge_count)
    throw RangeError();
  return m_edges[index];
}

std::vector<WordGraph::Node> & WordGraph::frame(int frame) {
  if (frame < 0 || frame >= m_frame_count)
    throw RangeError();
  return m_frames[frame];
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
