// Copyright (C) 2007  Vesa Siivola. 
// See licence.txt for the terms of distribution.

// Abstract class for common routines in reading arpa
#ifndef ARPAREADER_HH
#define ARPAREADER_HH

#include <cstdio>
#include <vector>
#include <string>

#include "Vocabulary.hh"

class ArpaReader {
public:
  ArpaReader(Vocabulary *voc) : m_lineno(0), m_read_order(0), m_gram_num(0), m_vocab(voc) {};
  void read_error();
  void read_header(FILE *, bool &, std::string &);
  bool next_gram(FILE *file, std::string &line, std::vector<int> &, float &, float &);
  std::vector<int> counts;

private:
  int m_lineno;
  int m_read_order;
  int m_gram_num;
  Vocabulary *m_vocab;

  // For efficiency of next_gram()
  std::vector<std::string> m_vec;

};

#endif /* ARPAREADER_HH */
