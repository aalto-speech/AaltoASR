#ifndef SIMPLEHASHCACHE_HH
#define SIMPLEHASHCACHE_HH

#define HASH_SIZE_MULTIPLIER 1.3

template <typename T>
class SimpleHashCache {
public:
  SimpleHashCache(void);
  inline bool insert(int key, T item, T *removed);
  inline bool find(int key, T *result);
  void set_max_items(int max);
  int get_num_items(void) { return num_items; }
  bool remove_last_item(T *removed);

private:
  void rehash(int new_max);
  inline int get_hash_index(int key) { return (key%hash_table_size); }
  
  class StoreType {
  public:
    T value;
    int key;
  };
  std::vector< StoreType > *hash_table;

  int hash_table_size;
  int num_items;
};

template<typename T>
SimpleHashCache<T>::SimpleHashCache(void)
{
  hash_table = NULL;
  hash_table_size = num_items = 0;
}

template<typename T>
void
SimpleHashCache<T>::rehash(int new_max)
{
  std::vector< StoreType > *new_hash_table = new std::vector< StoreType >;
  int new_size = (int)(HASH_SIZE_MULTIPLIER*new_max);
  new_hash_table->resize(new_size);
  for (int i = 0; i < new_size; i++)
    (*new_hash_table)[i].key = -1;
  if (hash_table != NULL)
  {
    // Copy at most new_max items
    // FIXME: Missing! We just delete the old hash.
    delete hash_table;
  }
  hash_table = new_hash_table;
  hash_table_size = new_size;
  num_items = 0;
}

// Returns true if an item is removed, in which case the removed item is
// stored to variable *removed (if it is != NULL).
template<typename T>
bool
SimpleHashCache<T>::insert(int key, T item, T *removed)
{
  int index = get_hash_index(key);
  bool rm = false;
  
  if ((*hash_table)[index].key != -1)
  {
    // Remove the item
    if (removed != NULL)
      *removed = (*hash_table)[index].value;
    rm = true;
  }
  else
    num_items++;
  (*hash_table)[index].value = item;
  (*hash_table)[index].key = key;
  return rm;
}

template<typename T>
bool
SimpleHashCache<T>::remove_last_item(T *removed)
{
  for (int i = 0; i < hash_table_size; i++)
  {
    if ((*hash_table)[i].key != -1)
    {
      if (removed != NULL)
        *removed = (*hash_table)[i].value;
      (*hash_table)[i].key = -1;
      num_items--;
      return true;
    }
  }
  return false;
}

template<typename T>
bool
SimpleHashCache<T>::find(int key, T *result)
{
  int index = get_hash_index(key);
  if ((*hash_table)[index].key != key)
      return false;

  *result = (*hash_table)[index].value;
  return true;
}


template<typename T>
void
SimpleHashCache<T>::set_max_items(int max)
{
  rehash(max);
}


#endif // SIMPLEHASHCACHE_HH
