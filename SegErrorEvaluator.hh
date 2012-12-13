#ifndef SEGERROREVALUATOR_HH
#define SEGERROREVALUATOR_HH

#include "HmmSet.hh"
#include "HmmNetBaumWelch.hh"
#include <functional>

namespace aku {

class SegErrorEvaluator : public HmmNetBaumWelch::CustomScoreQuery {
public:
  typedef enum {
    MWE, // Word/label error approximation, as defined by Povey
    MPE, // Phone error approximation, as defined by Povey
    MPFE_MONOPHONE_LABEL, // Correct monophone label
    MPFE_MONOPHONE_STATE, // Correct monophone label+state index
    MPFE_CONTEXT_LABEL, // Correct context phone label
    MPFE_PDF, // Correct PDF index
    MPFE_CONTEXT_PHONE_STATE, // A state of the correct CP
    MPFE_HYP_CONTEXT_PHONE_STATE, // A correct state in hypothesis CP
    MPE_SNFE // Symmetrically normalized frame error approximation
  } ErrorMode;

  /** Segmented lattice hierarchy used for evaluation in different modes
   *  (0 = frame lattice, 1 = sub-phone lattice, 2 = (context-)phone lattice
   *   3 = word lattice)
   *  MWE - 3
   *  MPE - 2
   *  MPFE_PDF - 0
   *  MPFE_CONTEXT_PHONE_STATE - 0
   *  MPFE_HYP_CONTEXT_PHONE_STATE - 0
   *  MPE_SNFE - 2
   */



  /// Helper class to iterate over reference arcs in frame order,
  /// starting from a particular frame
  class RefIterator {
  public:
    typedef std::forward_iterator_tag iterator_category;
    typedef int value_type;
    typedef std::ptrdiff_t difference_type;
    typedef const int& reference;
    typedef const int* pointer;

    RefIterator(SegErrorEvaluator &parent, std::vector<int> &arcs_in_frame,
                std::vector<int> &sorted_arcs, int from_frame);
    //RefIterator(const RefIterator& i);
    
    reference operator*() const { return m_in_frame?m_arcs_in_frame[m_index]:m_sorted_arcs[m_index]; }
    
    //pointer operator->() const { return (m_in_frame?&m_arcs_in_frame[m_index]:&m_sorted_arcs[i]); }
    
    RefIterator& operator++() {
      ++m_index;
      if (m_in_frame) {
        if (m_index >= (int)m_arcs_in_frame.size()) {
          m_in_frame = false;
          std::vector<int>::iterator it =
            std::lower_bound(m_sorted_arcs.begin(), m_sorted_arcs.end(),
                             m_frame+1,
                             SegErrorEvaluator::ref_arc_frame_const_compare(
                               m_parent));
          if (it == m_sorted_arcs.end())
            m_index = -1;
          else
            m_index = std::distance(m_sorted_arcs.begin(), it);
        }
      } else if (m_index >= (int)m_sorted_arcs.size()) {
        m_index = -1;
      }
      return *this;
    }
    
    RefIterator operator++(int) {
      RefIterator tmp(*this);
      ++*this;
      return tmp;
    }

    friend bool operator==(const RefIterator& x, const RefIterator& y) {
      if (x.m_index == -1 && y.m_index == -1)
        return true;
      return (x.m_frame == y.m_frame && x.m_in_frame == y.m_in_frame &&
              x.m_index == y.m_index);
   }

    friend bool operator!=(const RefIterator& x, const RefIterator& y) {
      if (x.m_index == -1 && y.m_index == -1)
        return false;
      return (x.m_index != y.m_index || x.m_frame != y.m_frame ||
              x.m_in_frame != y.m_in_frame);
   }


  private:
    SegErrorEvaluator &m_parent;
    std::vector<int> &m_arcs_in_frame;
    std::vector<int> &m_sorted_arcs;
    int m_frame;
    bool m_in_frame;
    int m_index;
  };

  friend class RefIterator;

  
public:
  SegErrorEvaluator() { m_first_frame = -1; m_error_mode = MPE; m_model = NULL; m_ignore_silence = false; m_binary_mpfe = true; m_silence_word = "_"; }

  // CustomScoreQuery interface
  virtual ~SegErrorEvaluator() { }
  virtual double custom_score(HmmNetBaumWelch::SegmentedLattice const *sl,
                              int arc_index);

  // Other public methods
  void set_mode(ErrorMode mode) { m_error_mode = mode; }
  void set_model(HmmSet *model) { m_model = model; }
  void set_ignore_silence(bool silence) { m_ignore_silence = silence; }
  void set_silence_word(const std::string &silence_word) { m_silence_word = silence_word; }
  void initialize_reference(HmmNetBaumWelch::SegmentedLattice const *ref_lattice);
  void reset(void);

  void add_snfe_ref_arc_error(
    int ref_arc_index, double error,
    std::vector< std::pair<int, double> > &snfe_ref_arcs);
  double get_minimum_snfe_error(
    int end_frame, std::vector< std::pair<int, double> > &snfe_ref_arcs);

  //int non_silence_frames(void) { return m_non_silence_frames; }
  //double non_silence_occupancy(void) { return m_non_silence_occupancy; }
  //int frames(void) { return (int)m_ref_segmentation.size(); }

private:
  std::string extract_center_phone(const std::string &label);
  std::string extract_sublabel(const std::string &label, int count);
  std::string extract_word(const std::string &label);

  RefIterator reference_iterator(int frame) {
    assert( frame >= m_first_frame && frame < m_last_frame );
    return RefIterator(*this, m_arcs_in_frame[frame - m_first_frame],
                       m_sorted_arcs, frame);
  }

  RefIterator reference_iterator_end(void) {
    return RefIterator(*this, m_sorted_arcs, m_sorted_arcs, -1);
  }

  struct ref_arc_frame_compare : public std::binary_function<int, int, bool> {
    ref_arc_frame_compare(SegErrorEvaluator const &parent) : m_parent(parent) { }
    bool operator()(int x, int y) {
      return m_parent.m_ref_lattice->nodes[m_parent.m_ref_lattice->arcs[x].source_node].frame < m_parent.m_ref_lattice->nodes[m_parent.m_ref_lattice->arcs[y].source_node].frame;
    }
  private:
    SegErrorEvaluator const &m_parent;
  };

  struct ref_arc_frame_const_compare : public std::binary_function<int, int, bool> {
    ref_arc_frame_const_compare(SegErrorEvaluator const &parent) : m_parent(parent) { }
    bool operator()(int x, int frame) {
      return m_parent.m_ref_lattice->nodes[m_parent.m_ref_lattice->arcs[x].source_node].frame < frame;
    }
  private:
    SegErrorEvaluator const &m_parent;
  };



private:
  int m_first_frame;
  int m_last_frame; //!< Exclusive

  ErrorMode m_error_mode;
  HmmSet *m_model;
  int m_non_silence_frames;
  double m_non_silence_occupancy;
  bool m_ignore_silence;
  std::string m_silence_word;
  bool m_binary_mpfe;


  HmmNetBaumWelch::SegmentedLattice const *m_ref_lattice;
  
  /** For each frame, the list of reference arcs that are active during that
   * frame.
   */
  std::vector< std::vector<int> > m_arcs_in_frame;
  /// Reference arc indices, sorted in ascending order by the starting frame
  std::vector<int> m_sorted_arcs;
};

}

#endif // SEGERROREVALUATOR_HH
