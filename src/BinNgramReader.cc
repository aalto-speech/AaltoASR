#include "BinNgramReader.hh"

void BinNgramReader::write(FILE *out, Ngram *ng, bool reflip) {
  fprintf (out,"%d\n",ng->size());
  for (int i=0; i<ng->size(); i++) {
    fprintf(out,"%s\n",ng->word(i).c_str());
  }
  fprintf(out,"%d %d\n",ng->order(),ng->m_nodes.size());
  if (Endian::big) {flip_endian(ng);} // Write little endian binary
  fwrite(&ng->m_nodes[0],ng->m_nodes.size()*sizeof(Ngram::Node),1,out);
  if (Endian::big && reflip) {flip_endian(ng);}// Restore to original endianess
}

void BinNgramReader::read(FILE *in, Ngram *ng) {
  char cbuf[1000];
  int si;

  fscanf(in,"%d\n",&si);
  for (int i=0; i<si; i++) {
    fgets(cbuf,1000,in);
    cbuf[strlen(cbuf)-1]='\0';
    ng->add(cbuf);
  }
  int m_nodes_size;
  fscanf(in,"%d %d\n",&ng->m_order,&m_nodes_size);
  ng->m_nodes.resize(m_nodes_size);

  fread(&ng->m_nodes[0],ng->m_nodes.size()*sizeof(Ngram::Node),1,in);
  if (Endian::big) {flip_endian(ng);} // Big-endian system here
}

void BinNgramReader::flip_endian(Ngram *ng) {
  char *ptr;
  for (int i=0; i<ng->m_nodes.size(); i++) {
    ptr=(char *) &(ng->m_nodes[i]); // Ugliness, we want the pointer
    Endian::convert(ptr,2);         // to increment one byte at time
    Endian::convert(ptr+2,4);
    Endian::convert(ptr+6,4);
    Endian::convert(ptr+10,4);
  }
}
