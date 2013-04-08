#ifndef REF_HH
#define REF_HH

#include <cstddef>  // NULL
#include <assert.h>

namespace ref {

  /** Automatic pointer holder for reference countable entity: links
   * on creating and unlinks on destruction. */
  template <class T>
  class Ptr {
  public:
    /** Default constructor. */
    Ptr() : m_ptr(NULL) { }

    /** Constructor. */
    Ptr(T *ptr) : m_ptr(ptr) 
    { 
      if (m_ptr != NULL)
        m_ptr->reference_count++;
    }

    /** Copy constructor. */
    Ptr(const Ptr<T> &a) : m_ptr(a.m_ptr) 
    { 
      if (m_ptr != NULL)
	m_ptr->reference_count++;
    }

    /** Assignment. */
    const Ptr<T> operator=(const Ptr<T> &a) 
    {
      set(a.m_ptr);
      return *this;
    }

    /** Destructor */
    ~Ptr() 
    { 
      release(m_ptr);
    }

    /** Set a new pointer and release possibly the old. */
    void set(T *ptr) 
    { 
      T *old_ptr = m_ptr;
      m_ptr = ptr;
      if (m_ptr != NULL)
	m_ptr->reference_count++;
      release(old_ptr);
    }

    /** Access the pointer. */
    T *ptr() { return m_ptr; }

    /** Access the pointer. */
    const T *ptr() const { return m_ptr; }

    /** Access the object data. */
    T *operator->() { return m_ptr; }

    /** Access the object data. */
    const T *operator->() const { return m_ptr; }

    /** Access the object. */
    T &ref() { return *m_ptr; }

    /** Conversion to object pointer. */
    operator T*() { return m_ptr; }

    /** Conversion to object. */
    operator T&() { return *m_ptr; }

  private:

    /** Release the pointer if set. */
    void release(T *ptr) 
    {
      if (ptr == NULL)
	return;

      assert(ptr->reference_count > 0);
      ptr->reference_count--;
      if (ptr->reference_count == 0)
	delete ptr;
    }

    T *m_ptr; //!< Pointer to the object.
  };  



  /** Automatic pointer holder for reference countable entity: links
   * on creating and unlinks on destruction.  NOTE: instead of
   * deleting, a destroy() method is called. */
  template <class T>
  class VPtr {
  public:
    /** Default constructor. */
    VPtr() : m_ptr(NULL) { }

    /** Constructor. */
    VPtr(T *ptr) : m_ptr(ptr) 
    { 
      m_ptr->reference_count++;
    }

    /** Copy constructor. */
    VPtr(const VPtr<T> &a) : m_ptr(a.m_ptr) 
    { 
      if (m_ptr != NULL)
	m_ptr->reference_count++;
    }

    /** Assignment. */
    const VPtr<T> operator=(const VPtr<T> &a) 
    {
      set(a.m_ptr);
      return *this;
    }

    /** Destructor */
    ~VPtr() 
    { 
      release(m_ptr);
    }

    /** Set a new pointer and release possibly the old. */
    void set(T *ptr) 
    { 
      T *old_ptr = m_ptr;
      m_ptr = ptr;
      if (m_ptr != NULL)
	m_ptr->reference_count++;
      release(old_ptr);
    }

    /** Access the pointer. */
    T *ptr() { return m_ptr; }

    /** Access the object data. */
    T *operator->() { return m_ptr; }

    /** Access the object. */
    T &ref() { return *m_ptr; }

    /** Conversion to object pointer. */
    operator T*() { return m_ptr; }

    /** Conversion to object. */
    operator T&() { return *m_ptr; }

  private:

    /** Release the pointer if set. */
    void release(T *ptr) 
    {
      if (ptr == NULL)
	return;

      assert(ptr->reference_count > 0);
      ptr->reference_count--;
      if (ptr->reference_count == 0)
	T::destroy(ptr);
    }

    T *m_ptr; //!< Pointer to the object.
  };  


};

#endif /* REF_HH */
