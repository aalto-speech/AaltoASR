#ifndef FIXEDARRAY_HH
#define FIXEDARRAY_HH

#include <errno.h>
#include <vector>

namespace fsalm {

  /** Fast templated array of fixed-width values with the same
   * interface as the bit-backed arrays.
   */
  template <typename T>
  class FixedArray {

  public:
    
    typedef int index_type;

    /** Construct an empty array. */
    FixedArray()
      : m_num_elems(0), m_capacity(0)
    {
    }

    /** Create an array with initial size.  The initial elements are
     * guaranteed to be zero.
     *
     * \param num_elems = initial number of elements in the array
     * \throw bit::invalid_argument one of the bit arguments is invalid
     */
    FixedArray(int num_elems)
      : m_num_elems(0), m_capacity(0)
    {
      resize(num_elems);
    }

    /** Clear the structure. */
    void clear() {
      resize(0);
    }

    /** Return the number of elements in the array. */
    int num_elems() const { 
      return m_num_elems;
    }

    /** Return the width of the array, i.e. the number of bits
     * allocated for each element. */
    unsigned int bits_per_elem() const {
      return sizeof(T) * 8;
    }

    /** Return the number of elements allocated for the underlying buffer. */
    int capacity() const {
      return m_capacity;
    }

    /** Change the number of elements in the array.  The values of the
     * possible new elements are guaranteed to be zero ONLY if the
     * array has never been resized smaller.  
     * \param num_elems = the new number of elements
     */
    void resize(int num_elems)
    {
      m_num_elems = num_elems;
      if (m_capacity < m_num_elems) {
        m_capacity = m_num_elems;
        m_buffer.resize(m_capacity);
      }
    }

    /** Change the capacity (number of allocated elements) of the
     * array.  Making the capacity smaller does nothing.
     * \param capacity = the number of elements to allocate
     */
    void reserve(int capacity)
    {
      if (capacity <= m_capacity)
        return;
      m_capacity = capacity;
      m_buffer.resize(m_capacity);
    }

    /** Change the width of the elements to given width (DISABLED IN
     * THIS TEMPLATE, JUST THROWS).
     */
    void set_width(unsigned int bits_per_elem)
    {
      throw Error("FixedArray16::set_width() not supported");
    }

    /** Set the value of an element.  
     *
     * \param elem = the index of the element to set
     * \param value = the value to set
     * \throw Error accessing outside the array
     */
    void set(int elem, T value)
    {
      if (elem >= m_num_elems)
        throw Error("bit::FixedArray::set(): out of range");
      m_buffer.at(elem) = value;
    }

    /** Set the value of an element growing the buffer if necessary.
     * If the current capacity is not enough, the capacity is doubled
     * (set to one from zero), and if that is not enough, the capacity
     * is set to (elem+1).
     *
     * \param elem = the index of the element to set
     * \param value = the value to set
     */
    void set_grow(int elem, T value)
    {
      if (elem >= m_num_elems) {
        if (elem >= m_capacity) {
          int new_capacity = m_capacity * 2;
          if (elem >= new_capacity)
            new_capacity = elem + 1;
          reserve(new_capacity);
        }
        resize(elem + 1);
      }
      m_buffer.at(elem) = value;
    }

    /** Set the value of an element growing and widening the buffer if
     * necessary (IN THIS TEMPLATE, THIS JUST CALLS set_grow()
     *
     * \param elem = the index of the element to set
     * \param value = the value to set
     */
    void set_grow_widen(int elem, T value)
    {
      set_grow(elem, value);
    }

    /** Return value of an element. 
     * \param elem = the index of the element
     * \return the value
     * \throw Error accessing outside the array
     */
    T get(int elem) const
    {
      if (elem >= m_num_elems)
        throw Error("bit::FixedArray::get(): out of range");
      return m_buffer.at(elem);
    }

    /** Const access the internal data buffer.  It is safe to write
     * and read the first data_len() bytes of the pointer.  Note that
     * set_grow(), set_grow_widen(), set_width(), resize() and
     * reserve() may invalidate the pointer. */
    const unsigned char *data() const
    {
      return (unsigned char*)&m_buffer.at(0);
    }

    /** Access the internal data buffer.  It is safe to write
     * and read the first data_len() bytes of the pointer.  Note that
     * set_grow(), set_grow_widen(), set_width(), resize() and
     * reserve() may invalidate the pointer. */
    unsigned char *data() 
    {
      return (unsigned char*)&m_buffer.at(0);
    }

    /** Return the safe access length of the internal data buffer
     * obtained with data() (i.e. minimum number of bytes needed to
     * store the elements of the array. */
    int data_len() const 
    {
      return m_num_elems * sizeof(T);
    }

    /** Write the array in file.  Note that capacity is not written
     * into the file.  When the array is read, the capacity is set to
     * number of elements.
     *
     * \param file = file stream to write to
     * \throw bit::io_error if write fails
     */
    void write(FILE *file) const
    {
      fputs("FXARRAY1:", file);
      fprintf(file, "%d:", m_num_elems);
      if (data_len() > 0) {
        size_t ret = fwrite(data(), data_len(), 1, file);
        if (ret != 1)
          throw Error(
            std::string("bit::FixedArray::write() fwrite failed: ") + 
            strerror(errno));
      }
    }

    /** Read the array from file.  Note that capacity is not stored in
     * the file.  The capacity is set to number of elements stored in
     * the file.
     *
     * \param file = file stream to read from
     * \throw bit::io_error if read fails
     */
    void read(FILE *file)
    {
      int version;
      int ret = fscanf(file, "FXARRAY%d:%d:", 
                       &version, &m_num_elems);
      if (ret != 2 || version != 1)
        throw Error(
          "bit::FixedArray::read() error while reading header");
      m_buffer.resize(m_num_elems);
      m_capacity = m_num_elems;
      if (data_len() > 0) {
        size_t ret = fread(data(), data_len(), 1, file);
        if (ret != 1) {
          if (ferror(file))
            throw Error(
              "bit::FixedArray::read() error while reading buffer");
          assert(feof(file));
          throw Error(
            "bit::FixedArray::read() eof while reading buffer");
        }
      }
    }

  private:

    /** Number of elements in the array. */
    int m_num_elems;

    /** Number of elements allocated for the underlying buffer.  When
     * values are written outside the buffer with set_grow() method,
     * the size of the underlying buffer is doubled. */
    int m_capacity;

    /** Buffer containing the array elements. */
    std::vector<T> m_buffer;

  };

};

#endif /* FIXEDARRAY_HH */
