// This is example code from http://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance
template<class T>
unsigned int levenshtein_distance(const T & s1, const T & s2) {
  const size_t len1 = s1.size(), len2 = s2.size();
  std::vector<unsigned int> col(len2+1), prevCol(len2+1);
  
  for (unsigned int i = 0; i < prevCol.size(); i++)
    prevCol[i] = i;
  for (unsigned int i = 0; i < len1; i++) {
    col[0] = i+1;
    for (unsigned int j = 0; j < len2; j++)
      col[j+1] = std::min( std::min( 1 + col[j], 1 + prevCol[1 + j]),
                           prevCol[j] + (s1[i]==s2[j] ? 0 : 1) );
    col.swap(prevCol);
  }
  return prevCol[len2];
}

// Basically, a copy of the above function with an additional cutoff parameter
// FIXME: What would the performance penalty be if the two functions were merged into one?
template<class T>
unsigned int levenshtein_with_cutoff(const T & s1, const T & s2, int max_distance) {
  const size_t len1 = s1.size(), len2 = s2.size();
  std::vector<unsigned int> col(len2+1), prevCol(len2+1);
  
  for (unsigned int i = 0; i < prevCol.size(); i++)
    prevCol[i] = i;
  for (unsigned int i = 0; i < len1; i++) {
    unsigned int max_dist = 99999999; // FIXME: From get the max value from limits.h
    col[0] = i+1;
    for (unsigned int j = 0; j < len2; j++) {
       unsigned int cur_dist = std::min( std::min( 1 + col[j], 1 + prevCol[1 + j]),
                                prevCol[j] + (s1[i]==s2[j] ? 0 : 1) );
       max_dist = std::min(cur_dist, max_dist);
       col[j+1] = cur_dist;
    }
    if (max_dist >= max_distance) return max_dist;
    col.swap(prevCol);
  }
  return prevCol[len2];
}
