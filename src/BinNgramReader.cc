#include "BinNgramReader.hh"

void 
BinNgramReader::write(FILE *out, Ngram *ng, bool reflip) 
{
  fprintf (out, "cis-binlm1\n");
  fprintf (out, "%d\n", ng->size());
  for (int i=0; i<ng->size(); i++) {
    fprintf(out, "%s\n", ng->word(i).c_str());
  }
  fprintf(out, "%d %d\n", ng->order(), ng->m_nodes.size());
  if (Endian::big) {flip_endian(ng);} // Write little endian binary
  fwrite(&ng->m_nodes[0], ng->m_nodes.size()*sizeof(Ngram::Node), 1, out);
  if (Endian::big && reflip) {flip_endian(ng);}// Restore to original endianess
}

void 
BinNgramReader::read(FILE *in, Ngram *ng) 
{
  char cbuf[4096];
  int words;
  char *ptr;

  // Read the header
  ptr = fgets(cbuf, 4096, in);
  if (strcmp(cbuf, "cis-binlm1\n") != 0) {
    fprintf(stderr, "BinNgramReader::read(): invalid file format\n");
    exit(1);
  }

  // Read the number of words
  ptr = fgets(cbuf, 4096, in);
  if (!ptr) {
    fprintf(stderr, "BinNgramReader::read(): unexpected end of file\n");
    exit(1);
  }
  words = atoi(cbuf);
  if (words < 1) {
    fprintf(stderr, "BinNgramReader::read(): invalid number of words: %s\n", 
	    cbuf);
    exit(1);
  }
  
  // Read the vocabulary
  for (int i=0; i < words; i++) {
    ptr = fgets(cbuf, 4096, in);
    if (!ptr) {
      fprintf(stderr, "BinNgramReader::read(): read error while reading vocabulary\n");
      exit(1);
    }
    cbuf[strlen(cbuf) - 1]='\0';
    ng->add(cbuf);
  }

  // Read nodes
  int m_nodes_size;
  fscanf(in, "%d %d\n", &ng->m_order, &m_nodes_size);
  ng->m_nodes.resize(m_nodes_size);

  size_t block_size = ng->m_nodes.size() * sizeof(Ngram::Node);
  size_t blocks_read = fread(&ng->m_nodes[0], block_size, 1, in);
  if (blocks_read != 1) {
      fprintf(stderr, "BinNgramReader::read(): read error while reading ngrams\n");
      exit(1);
  }
  if (Endian::big) 
    flip_endian(ng);
}

void BinNgramReader::flip_endian(Ngram *ng) 
{
  for (int i=0; i<ng->m_nodes.size(); i++) {
    Endian::convert(&ng->m_nodes[i].word, 4);
    Endian::convert(&ng->m_nodes[i].log_prob, 4);
    Endian::convert(&ng->m_nodes[i].back_off, 4);
    Endian::convert(&ng->m_nodes[i].first, 4);
  }
}
