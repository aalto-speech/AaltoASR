#ifndef REGCLASSTREE_H_
#define REGCLASSTREE_H_

#include "HmmSet.hh"
#include <map>
#include <set>
#include <vector>


namespace aku {

/**
 * Container for the 'atomic' units in the Regression tree.
 */
class RegClassTree
{

public:
  class Unit
  {
  protected:
    Vector m_mean;
    Matrix m_covar;
    double m_occ;

    std::set<int> m_pdf_indices;

  public:
    virtual void get_mean(Vector& v) const { v.copy(m_mean); }
    virtual void get_covar(Matrix& m) const { m.copy(m_covar); }
    virtual double get_occ() const { return m_occ; }

    virtual const std::string& get_identifier() = 0;
    virtual void calculate_statistics(HmmSet *model) = 0;

    virtual void get_pdf_indices(HmmSet *model, std::set<int> &v) {
      if(m_pdf_indices.size() == 0) gather_pdf_indices(model);
      if(m_pdf_indices.size() > 0) v.insert(m_pdf_indices.begin(), m_pdf_indices.end());
    }

    virtual void gather_pdf_indices(HmmSet *model) = 0;
    virtual void initialize_from_model(HmmSet *model) = 0;
    static std::vector<Unit*> get_all_components(HmmSet *model);


    Unit() { }
    virtual ~Unit() { }

  };

  /**
   * Tree unit for phonemes (derived by taking all the center phones of all Hmm's )
   */
  class UnitPhoneme : public Unit
  {
  private:
    std::string m_phone;
    std::vector<Hmm> m_hmms;

  public:
    static std::vector<Unit*> get_all_components(HmmSet *model);
    static void get_gaussians(HmmSet *model, const std::vector<std::string> &phones, std::set<int> &gaussians);
    virtual const std::string& get_identifier() { return m_phone; }
    virtual void calculate_statistics(HmmSet *model);

    virtual void gather_pdf_indices(HmmSet *model);

    virtual void initialize_from_model(HmmSet *model);

    UnitPhoneme() { }
    UnitPhoneme(std::string phone) : m_phone(phone) {  }
    virtual ~UnitPhoneme() { }
  };

  /**
   * Warning: don't use this class if gaussians are shared between mixtures!
   */
  class UnitMixture : public Unit
  {
  private:
    std::string m_id;
    Mixture *m_mixture;
  public:
    static std::vector<Unit*> get_all_components(HmmSet *model);
    static void get_gaussians(HmmSet *model, const std::vector<std::string> &mixtures, std::set<int> &gaussians);
    virtual const std::string& get_identifier() { return m_id; }
    virtual void calculate_statistics(HmmSet *model);
    virtual ~UnitMixture() { }
    virtual void initialize_from_model(HmmSet *model);
    virtual void gather_pdf_indices(HmmSet *model);

    UnitMixture(std::string id) : m_id(id) {}
    UnitMixture(int id, Mixture* mix) : m_mixture(mix) {
      std::stringstream out;
      out << id;
      m_id = out.str();
    }
  };

  class UnitGaussian : public Unit
  {
  private:
    std::string m_id;
    Gaussian* m_g;

  public:
    static std::vector<Unit*> get_all_components(HmmSet *model);
    static void get_gaussians(HmmSet *model, const std::vector<std::string> &gausstrings, std::set<int> &gaussians);
    virtual const std::string& get_identifier() { return m_id; }
    virtual void calculate_statistics(HmmSet *model);
    virtual void initialize_from_model(HmmSet *model);
    virtual void gather_pdf_indices(HmmSet *model);

    UnitGaussian(std::string id) : m_id(id) { }
    UnitGaussian(int id, Gaussian* g) : m_g(g) {
          std::stringstream out;
          out << id;
          m_id = out.str();
        }
    virtual ~UnitGaussian() { }
  };

  /**
   * UnitGlobal is used for a global transform. This unit always contains all pdf's and does not split them. Matches with the UNIT_NO Regression tree.
   *
   */
  class UnitGlobal : public Unit
  {
  private:
    std::string m_id;

  public:
    static std::vector<Unit*> get_all_components(HmmSet *model);
    static void get_gaussians(HmmSet *model, std::set<int> &gaussians);

    virtual const std::string& get_identifier() { return m_id; }
    virtual void calculate_statistics(HmmSet *model) { m_mean = 0; m_covar = 0; m_occ = 1; }
    virtual void initialize_from_model(HmmSet *model) {}
    virtual void gather_pdf_indices(HmmSet *model);

    UnitGlobal() : m_id("") { }
    virtual ~UnitGlobal() {}
  };
  /**
   * A node in the regression class tree.
   * Important: bool terminal_node is indicating or the children nodes are allocated. This must be always correct to avoid access of null pointers and memory leaks
   */
  class Node
  {
  public:
    std::vector<Unit*> m_components;
    Node *m_c1;
    Node *m_c2;

    /**
     * The mean of this node
     */
    Vector m_mean;

    /**
     * The covariance of this node
     */
    Matrix m_covar;

    /**
     * Binary index (like a heap). Root of a tree has index 1
     */
    const int m_index;

    /**
     * indicates or this node has children
     */
    bool m_terminal_node;

    /**
     * Total occupancy of node
     */
    double m_total_occ;

    /**
     * score of node ( score = sum_{components} (distance(nodemean, componentmean) * componentoccupancy)
     */
    double m_score;


    double get_distance(Unit *rcu) const;

    void update_score_mean();
    void update_covar();
    void write(std::ostream *out, bool print_empty_nodes = false) const;

    void get_terminal_child_nodes(std::vector<Node*> &v);
    void get_pdf_indices(HmmSet *model, std::set<int> &v);

    std::vector<std::string> get_all_unit_identifiers() {
      std::vector<std::string> identifiers;
      get_all_unit_identifiers(identifiers);

      return identifiers;
    }

    void get_all_unit_identifiers(std::vector<std::string> &identifiers) {
      for(unsigned int i = 0; i < m_components.size(); ++i)
        identifiers.push_back(m_components[i]->get_identifier());
      if(!m_terminal_node) {
        m_c1->get_all_unit_identifiers(identifiers);
        m_c2->get_all_unit_identifiers(identifiers);
      }
    }

    Node(int index, int vector_size) : m_index(index), m_terminal_node(true)
    {
      m_mean.resize(vector_size);
      m_covar.resize(vector_size, vector_size);
    }
    ~Node() {
      if (!m_terminal_node) {
        delete m_c1;
        delete m_c2;
      }
      for (unsigned int i = 0; i < m_components.size(); ++i)
        delete m_components[i];
    }

    /**
     * Struct used for ordering RegClassNodes (based on score)
     */
    struct RegClassCmp
    {
      bool operator()(Node* r1, Node* r2) { return r1->m_score < r2->m_score; }
    };

  };
public:
  enum UnitMode
  {
    UNIT_NO = 0, UNIT_PHONE = 1, UNIT_MIX = 2, UNIT_GAUSSIAN = 3
  };
private:
  Node* m_root;
  UnitMode m_unit_mode;

  int m_dim;

  std::vector<double> m_state_frame_count;

  void split_node(Node *parent, double iter_treshold = 0.00001, double perturbation = 0.2);

public:
  /**
   * This method returns a pointer to the node with the corresponding index in the binary tree
   * Creates dynamically the nodes if not existent yet.
   *
   * root must exist before this method is called.
   */
  Node* get_node(int index);

public:
  RegClassTree() { }
  RegClassTree(UnitMode um) : m_unit_mode(um) { }
  virtual ~RegClassTree();
  void initialize_root_node(HmmSet *model);
  void build_tree(int n_terminals);
  void set_unit_mode(UnitMode um) { m_unit_mode = um; }
  UnitMode get_unit_mode() const { return m_unit_mode; }
  void write(std::ostream *out) const;
  void read(std::istream *in, HmmSet *model);

  void get_terminal_nodes(std::vector<Node*> &v);

  Node* get_root_node() const { return m_root; }
};

class PDFGroupModule {
public:
  virtual void merge(PDFGroupModule *pgm) = 0;
  virtual double get_frame_count() = 0;
  PDFGroupModule() {}
  virtual ~PDFGroupModule() {}
};

template<class T>
  class TreeToModuleMap
  {
    //TODO: make this a vector
    std::map<int, T*> m_pdf_to_t;
    std::map<int, T*> m_node_to_t;

    RegClassTree *m_tree;
    HmmSet *m_model;

  public:
    TreeToModuleMap(RegClassTree *tree, HmmSet *model);

    T* get_module(int pdf_id) { return m_pdf_to_t[pdf_id]; }

    /**
     * Merge the module instances with not enough frames
     */
    void merge_modules(double min_frames);

    RegClassTree::UnitMode get_unit_mode() {
      return m_tree->get_unit_mode();
    }


    std::map<std::vector<std::string>, T*> get_comps_to_t_map();

    /**
     * Delete all T instances created by this map
     * Remove all mappings
     */
    void clear();

    ~TreeToModuleMap() {
      clear(); }

  private:
    void check_node_for_merge(RegClassTree::Node *node, double min_frames);
    void merge_node(RegClassTree::Node *node);
    double get_num_frames(RegClassTree::Node *node);
    bool has_module(RegClassTree::Node *node) const;
  };

template<class T>
  TreeToModuleMap<T>::TreeToModuleMap(RegClassTree *tree, HmmSet *model) :
    m_tree(tree), m_model(model)
  {

    std::vector<RegClassTree::Node*> nodes;
    std::set<int> pdfs;

    tree->get_terminal_nodes(nodes);


    for (unsigned int i = 0; i < nodes.size(); ++i) {
      nodes[i]->get_pdf_indices(m_model, pdfs);

      m_node_to_t[nodes[i]->m_index] = new T(model);

      std::set<int>::iterator it;
      for (it = pdfs.begin(); it != pdfs.end(); ++it)
        m_pdf_to_t[*it] = m_node_to_t[nodes[i]->m_index];

      pdfs.clear();
    }
  }

template<class T>
  void
  TreeToModuleMap<T>::merge_modules(double min_frames)
  {
    check_node_for_merge(m_tree->get_root_node(), min_frames);
  }

template<class T>
  void
  TreeToModuleMap<T>::check_node_for_merge(RegClassTree::Node *node,
      double min_frames)
  {
    if (!node->m_terminal_node && !has_module(node)) {

      if (get_num_frames(node->m_c1) < min_frames || get_num_frames(node->m_c2)
          < min_frames) {
        merge_node(node);
      }
      else {
        check_node_for_merge(node->m_c1, min_frames);
        check_node_for_merge(node->m_c2, min_frames);
      }
    }
  }

template<class T>
  bool
  TreeToModuleMap<T>::has_module(RegClassTree::Node *node) const
  {
    return (m_node_to_t.find(node->m_index) != m_node_to_t.end());
  }

template<class T>
  void
  TreeToModuleMap<T>::merge_node(RegClassTree::Node *node)
  {
    if (!has_module(node->m_c1)) merge_node(node->m_c1);
    if (!has_module(node->m_c2)) merge_node(node->m_c2);

    T *mod1 = m_node_to_t[node->m_c1->m_index];
    PDFGroupModule *mod2 = m_node_to_t[node->m_c2->m_index];

    mod1->merge(mod2);
    delete mod2;

    m_node_to_t.erase(node->m_c1->m_index);
    m_node_to_t.erase(node->m_c2->m_index);

    m_node_to_t[node->m_index] = mod1;
  }

template<class T>
  double
  TreeToModuleMap<T>::get_num_frames(RegClassTree::Node *node)
  {
    if (has_module(node)) {
      PDFGroupModule *mod = m_node_to_t[node->m_index];
      return mod->get_frame_count();
    }
    else {
      return get_num_frames(node->m_c1) + get_num_frames(node->m_c2);
    }
  }


template<class T>
  void
  TreeToModuleMap<T>::clear()
  {
    typename std::map<int, T*>::iterator it;
    for (it = m_node_to_t.begin(); it != m_node_to_t.end(); it++)
      delete it->second;


    m_node_to_t.clear();
    m_pdf_to_t.clear();
  }

template<class T>
  std::map<std::vector<std::string>, T*>
  TreeToModuleMap<T>::get_comps_to_t_map()
  {
    std::map<std::vector<std::string>, T*> comps_to_t;

    typename std::map<int, T*>::iterator it;

    for (it = m_node_to_t.begin(); it != m_node_to_t.end(); ++it)
      comps_to_t[m_tree->get_node(it->first)->get_all_unit_identifiers()]
          = it->second;

    return comps_to_t;

  }

}

#endif /* REGCLASSTREE_H_ */
