#ifndef CLUSTERMAP_HH
#define CLUSTERMAP_HH

//#include "NgramCounts.hh"
#include "TreeGram.hh"

const int MAX_WLEN=1000;

template <typename KT, typename CT>
class ClusterMap : public Vocabulary {
public:
  void init_order(const int order, const int size);
  inline int num_clusters(const int order);
  //sikMatrix<KT, CT> *transform_counts(sikMatrix<KT, CT> *mat);
  inline void change_cluster(const int order, const KT newc, const KT word);
  inline int get_cluster(const int order, KT w);
  inline int get_cluster2(const int order, const KT cl);  
  int read(FILE *in, const int ord, int read_lines);
  void write(FILE *out);
  void wv2cv(std::vector<KT> &v); 
  void wg2cg(TreeGram::Gram &g);

/*  inline void wv2cv(std::vector<unsigned short> &v) {
    std::vector<int> v2;
    wv2cv(v2);
    for (int i=0;i<v.size();i++) v[i]=v2[i];
  }
*/

private:
  std::vector<std::vector <KT> > m_map;
  std::vector<KT> m_num_cl;
};

template <typename KT, typename CT>
int ClusterMap<KT,CT>::num_clusters(const int order) {
  return(m_num_cl.at(order));
}

template <typename KT, typename CT>
void ClusterMap<KT,CT>::change_cluster(const int order, const KT newc, const KT word) {
  // Should assert things here
  m_map[order][word]=newc;
  if (newc>=m_num_cl[order]) m_num_cl[order]=newc+1;
}

template <typename KT, typename CT>
int ClusterMap<KT,CT>::get_cluster(const int order, KT w) {
  for (int i=1;i<=order;i++) {
    w=m_map[i][w];
  }
  return(w);
}

template <typename KT, typename CT>
int ClusterMap<KT, CT>::get_cluster2(const int order, const KT cl) {
  return(m_map[order][cl]);
} 

#include "ClusterMap_tmpl.hh"
#endif
