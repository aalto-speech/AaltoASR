#include "BinNgramReader.hh"

#include "tools.hh"

void 
BinNgramReader::write(FILE *file, Ngram *ng, bool reflip) 
{
  fprintf (file, "cis-binlm1\n");
  fprintf (file, "%d\n", ng->num_words());
  for (int i=0; i<ng->num_words(); i++)
    fprintf(file, "%s\n", ng->word(i).c_str());
  fprintf(file, "%d %d\n", ng->order(), ng->m_nodes.size());

  // Write little endian binary
  if (Endian::big) 
    flip_endian(ng); 

  fwrite(&ng->m_nodes[0], ng->m_nodes.size() * sizeof(Ngram::Node), 1, file);
  
  // Restore to original endianess
  if (Endian::big && reflip)
    flip_endian(ng);
}

void 
BinNgramReader::read(FILE *file, Ngram *ng) 
{
  std::string line;
  int words;
  std::string format_str("cis-binlm1\n");
  bool ret;

  // Read the header
  ret = read_string(&line, format_str.length(), file);
  if (!ret || line != format_str) {
    fprintf(stderr, "BinNgramReader::read(): invalid file format\n");
    exit(1);
  }

  // Read the number of words
  if (!read_line(&line, file)) {
    fprintf(stderr, "BinNgramReader::read(): unexpected end of file\n");
    exit(1);
  }
  words = atoi(line.c_str());
  if (words < 1) {
    fprintf(stderr, "BinNgramReader::read(): invalid number of words: %s\n", 
	    line.c_str());
    exit(1);
  }
  
  // Read the vocabulary
  for (int i=0; i < words; i++) {
    if (!read_line(&line, file)) {
      fprintf(stderr, "BinNgramReader::read(): read error while reading vocabulary\n");
      exit(1);
    }
    chomp(&line);
    ng->add_word(line);
  }

  // Read the order and the number of nodes
  int m_nodes_size;
  fscanf(file, "%d %d\n", &ng->m_order, &m_nodes_size);
  ng->m_nodes.resize(m_nodes_size);

  // Read the nodes
  size_t block_size = ng->m_nodes.size() * sizeof(Ngram::Node);
  size_t blocks_read = fread(&ng->m_nodes[0], block_size, 1, file);
  if (blocks_read != 1) {
      fprintf(stderr, "BinNgramReader::read(): read error while reading ngrams\n");
      exit(1);
  }

  if (Endian::big) 
    flip_endian(ng);
}

void 
BinNgramReader::flip_endian(Ngram *ng) 
{
  for (int i=0; i<ng->m_nodes.size(); i++) {
    Endian::convert(&ng->m_nodes[i].word, 4);
    Endian::convert(&ng->m_nodes[i].log_prob, 4);
    Endian::convert(&ng->m_nodes[i].back_off, 4);
    Endian::convert(&ng->m_nodes[i].first, 4);
  }
}
