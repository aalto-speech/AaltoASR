#include "WordGraph.hh"

#define READ_MODE "r"
#define LINE_SIZE 256

WordGraph::WordGraph(FILE *file, Vocabulary *vocab) {
  this->file = file;
  if (this->file == NULL) (this->ok_flag = false);
  else (this->ok_flag = true);
  this->read_mode = NODES;
  this->node_table = NULL;
  this->vocab = vocab;
}

WordGraph::~WordGraph() {

  for (int i = 0; i < this->node_count; i++)
    delete this->node_table[i];

  delete this->node_table;

  for (int j = 0; j < this->edge_count; j++)
    delete edge_vector[j];
}

int WordGraph::read() {

  int index;
  char line[LINE_SIZE];
  char word[LINE_SIZE];
  WordGraph::Node *temp_node = NULL;
  WordGraph::Edge *temp_edge = NULL;

  while(this->ok_flag && !feof(this->file) && fgets(line, LINE_SIZE, this->file) != NULL) {

    // discard comment & empty lines
    if (line[0] == '#' || line[0] == '\n');


    // node/edge parsing
    else if (isdigit(line[0])) {
      
      // node parsing
      if (this->read_mode == NODES) {

	temp_node = new WordGraph::Node;
	
	if (sscanf(line, "%i%*[ ]%[^ ]%*[ ]%u%*u%*u", 
		   &index, word, &temp_node->frame) != 3) 
	  this->ok_flag = false;
	
	temp_node->word_id = this->vocab->index(word);
	this->node_table[index] = temp_node;
      }
      
      // edge parsing
      else if (this->read_mode == EDGES) {

	temp_edge = new WordGraph::Edge;
	
	if (sscanf(line, "%u%*[ ]%u%*[ ]%f",
		   &index, &temp_edge->target_node, &temp_edge->ac_log_prob) != 3)
	  this->ok_flag = false;
	
	this->node_table[index]->edges.push_back(this->edge_count);
	this->edge_vector.push_back(temp_edge);
	this->edge_count++;	
      }      
    }

    // command parsing
    else if (isalpha(line[0])) {

      // 'Frames <frame_count>' keyword
      if (strncmp(line, "Frames", 6) == 0) {
	sscanf(line, "%*[^ ] %u", &this->frame_count);
      }

      // 'Nodes <node_count>' keyword
      else if (strncmp(line, "Nodes", 5) == 0) {
	sscanf(line, "%*[^ ] %u", &this->node_count);
	this->read_mode = NODES;
	this->node_table = new WordGraph::Node*[this->node_count];
      }

      // 'Initial <init_count>' keyword
      else if (strncmp(line, "Initial", 5) == 0) {
      }

      // 'Final <final_count>' keyword
      else if (strncmp(line, "Final", 5) == 0) {
      }

      // 'BestSegAscr <?_number>' keyword
      else if (strncmp(line, "BestSegAscr", 5) == 0) {
	sscanf(line, "%*[^ ] %u", &this->best_seg_ascr);
      }

      // 'Edges' keyword
      else if (strncmp(line, "Edges", 5) == 0) {
	this->read_mode = EDGES;
      }
    }
    
    // error
    else {
      this->ok_flag = false;
    }
  } 
  
  if (this->file != NULL)
    if (!fclose(this->file)) this->ok_flag = false;
  
  if (this->ok_flag) return 1;
  else return 0;
}

WordGraph::Node & WordGraph::node(int index) {
  if (index < this->node_count)
    return (*this->node_table[index]);
  else return *(new WordGraph::Node);
}

WordGraph::Edge & WordGraph::edge(int index) {
  if (index < this->edge_count)
    return (*this->edge_vector[index]);
  else return *(new WordGraph::Edge);
}

int WordGraph::nodes() {
  return this->node_count;
}

int WordGraph::edges() {
  return this->edge_count;
}

std::vector<WordGraph::Node*> WordGraph::frame_nodes(int frame) {

  std::vector<WordGraph::Node*> node_vector = *(new std::vector<WordGraph::Node*>);

  for (int i = 0; i < this->node_count; i++) {
    if (this->node_table[i]->frame == frame) 
      node_vector.push_back(this->node_table[i]);
  }
  return node_vector;
}
