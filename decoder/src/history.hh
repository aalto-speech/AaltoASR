#ifndef HISTORY_HH
#define HISTORY_HH

#include <cstddef>  // NULL
#include <cassert>

namespace hist {

  /** Decrease the reference count of the structure and unlink the
   * structures recursively if reference count becomes zero. */ 
  template <class T>
  //void unlink(T *orig, void (release_func)(T*)=NULL) 
  void unlink(T *orig, std::vector<T*> *pool=NULL) 
  {
    T *t = orig;
    while (1) {
      if (t == NULL)
	return;

      assert(t->reference_count >= 0);
      if (t->reference_count > 1)
	break;

      T *previous = t->previous;
      if (pool) {
        pool->push_back(t);
      } else {
        delete t;
      }
      t = previous;
    }
    assert(t->reference_count > 0 && t->reference_count < 1000000);
    t->reference_count--;
  }

  /** Increase the reference count of the structure. */
  template <class T>
  void link(T *t) {
    assert(t->reference_count >= 0 && t->reference_count < 1000000);
    t->reference_count++;
  }

  /** Automatic holder for reference countable entity: links on
   * creating and unlinks on destruction. */
  template <class T>
  class Auto {
  public:
    /** Default constructor. */
    Auto() : m_obj(NULL), m_pool(NULL) { }

    /** Constructor. */
    Auto(T *obj) : m_obj(obj), m_pool(NULL) { link(obj); }

    Auto(T *obj, std::vector<T *> *pool) : m_obj(obj), m_pool(pool) { link(obj); }

    /** Destructor */
    ~Auto() { if (m_obj) unlink(m_obj, m_pool); }

    /** Link */
    void adopt(T *obj) { 
      if (m_obj != NULL)
	hist::unlink(m_obj);
      m_obj = obj;
      hist::link(obj);
    }

    /** Link */
    void adopt(T *obj, std::vector<T *> *pool) { 
      if (m_obj != NULL)
	hist::unlink(m_obj, pool);
      m_obj = obj;
      m_pool = pool;
      hist::link(obj);
    }


  private:
    T *m_obj; //!< Pointer to the object.
    std::vector<T *> *m_pool;
  };
  
};

#endif /* HISTORY_HH */
