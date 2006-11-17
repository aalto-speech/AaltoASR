#ifndef CLUSTERMAP_HH
#define CLUSTERMAP_HH

#include "TreeGram.hh"

template <typename KT>
class ClusterMap : public Vocabulary {
public:
  virtual ~ClusterMap() {}
  inline int order() {return m_map.size()-1;}
  virtual bool init_order(const int order);
  inline void set_cluster(const int order, const KT lowcl, const KT oricl, 
			  const KT newc);
  inline KT get_cluster(const int order, KT w);
  inline KT get_cluster2(const int order, const KT cl);  
  virtual int read(FILE *in, const int ord, int read_lines);
  virtual void read_more(FILE *in);
  virtual void write(FILE *out, int order=0);
  void wv2cv(std::vector<KT> &v); 
  void wg2cg(TreeGram::Gram &g);
  inline int num_clusters(const int o) {return(m_num_cl[o]);}
  void read_error(const int line, const std::string &text);
  virtual inline KT get_fcluster(const int order, const KT w) {return(w);}
  virtual inline KT get_fcluster2(const int order, const KT cl) {return(cl);}
  virtual inline float get_full_emprob(const int order, KT w) {return(0.0);}  
  virtual inline int num_fclusters(const int o) {return(num_words());}

// For prune.cc
//  void init_backwards_map();

protected:
  std::vector<std::vector <KT> > m_map;
  std::vector<int> m_num_cl;

//For prune.cc
//  std::vector<std::vector<std::vector<std::vector<int> > > > backmap;
};

template <typename KT>
class ClusterFMap : public ClusterMap<KT> {
public:
  bool init_order(const int order);
  int read(FILE *in, const int ord, int read_lines);
  void write(FILE *out);

  inline void set_fcluster(const int order, const KT lowcl, const KT oricl, const KT newcl);
  inline KT get_fcluster(const int order, KT w);
  inline KT get_fcluster2(const int order, const KT cl);  
  inline float get_full_emprob(const int order, KT w);  
  inline float get_fprob(const int order, KT w);  
  inline float get_fprob2(const int order, KT w);  
  inline void set_fprob(const int order, KT w, const float pr);  
  inline int num_fclusters(const int o) {
    return(m_num_fcl[o]);
  }

  struct key_prob {
    inline key_prob() : key(0), lprob(0) {}
    KT key;
    float lprob;

    bool operator<(const key_prob &other) {
      return(key < other.key);
    }
  };

private:
  std::vector<std::vector <key_prob> > m_fmap; // Forward maps
  struct sort_key_prob {
    bool operator()(const key_prob &x, const key_prob &y) { return x.key < y.key; }
  };
  std::vector<int> m_num_fcl;
};

#include "ClusterMap_tmpl.hh"
#endif
