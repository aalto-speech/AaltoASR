template <typename KT, typename CT>
void ClusterMap<KT, CT>::init_order(const int order, const int vocabsize) {
  int old_order=m_map.size()-1;
  m_map.resize(order+1);
  m_num_cl.resize(order+1);

  if (old_order<order) {
    m_map[order].resize(vocabsize,0);
    m_num_cl[order]=1;
  }

  if (order!=1) return;
  for (KT i=0;i<num_words();i++) {
    m_map[1][i]=i;
  }
  m_num_cl[1]=num_words();
}

#if 0
template <typename KT, typename CT>
sikMatrix<KT, CT> *ClusterMap<KT, CT>::transform_counts(sikMatrix<KT, CT> *oldmat) {
  /* Transform counts, use the n-1 transform for the last counts n */
  CT val;
  std::vector<KT> v(oldmat->dims);

  for (int i=1;i<oldmat->dims;i++) {
    v[i]=m_num_cl[oldmat->dims-i];
  }
  v[0]=m_num_cl[oldmat->dims-1];

  sikMatrix<KT, CT> *mat=new sikMatrix<KT, CT>(oldmat->dims, oldmat->m->hashsize, 0);

  oldmat->stepthrough(true, &v[0], &val);
  while (oldmat->stepthrough(false, &v[0], &val)) {
    v[0]=get_cluster(mat->dims-1,v[0]);
    for (int i=1;i<mat->dims;i++) {
      v[i]=get_cluster(mat->dims-i,v[i]);
    }
    mat->increment(&v[0],val);
  }
  return(mat);
}
#endif

template <typename KT, typename CT>
void ClusterMap<KT,CT>::write(FILE *out) {
  fprintf(out,"\\clustermap %d\n",m_map.size()-1);
  fprintf(out,"vocab=%d\n",num_words());
  for (int i=1;i<m_map.size();i++) {
    fprintf(out,"\\%d-ords\n",i);
      for (int j=0;j<num_words();j++) {
	fprintf(out,"%s %d\n",word(j).c_str(),get_cluster(i,j));
      }
  }
  fprintf(out,"\\endcl\n\n");
}

template <typename KT, typename CT>
int ClusterMap<KT,CT>::read(FILE *in, const int ord, int read_lines) {
  // Assuming that "\\clustermap %d" has already been read
  int vocabsize,o2;
  char cbuf[MAX_WLEN];

  m_map.resize(ord+1);
  read_lines++;
  if (fscanf(in,"vocab=%d\n",&vocabsize)!=1) {
    fprintf(stderr,"Error reading clustermap, line %d. Exit.\n",read_lines);
    exit(-1);
  }

  for (int o=1;o<=ord;o++) {
    m_map[o].resize(vocabsize);
    read_lines++;
    if (fscanf(in,"\\%d-ords\n",&o2)==0) {
      fprintf(stderr,"Error reading clustermap, line %d. Exit.\n",read_lines);
      exit(-1);
    }
    assert(o2=o);

    for (int x=0;x<vocabsize;x++) {
      int c;
      read_lines++;
      if (fscanf(in,"%s %d\n",cbuf,&c)==-1) {
	fprintf(stderr,"Error2 reading clustermap, line %d. Exit.\n",
		read_lines);
	exit(-1);
      }

      int c_prev;
      if (o==1) {
	add_word(cbuf);
	c_prev=word_index(cbuf);
      } else { 
      /* Check to which cluster the word was mapped in previous order,
	 put this index into m_map. save format has redundancy */
	c_prev=m_map[o-1][word_index(cbuf)]; 
      }
      m_map[o][c_prev]=c;
    }
  }
  read_lines++;
  if (fscanf(in,"\\endcl\n")!=0) {
    fprintf(stderr,"Error3 reading clustermap, line %d. Exit.\n",read_lines);
    exit(-1);
  }
  return(read_lines);
}

template <typename KT,typename CT>
void ClusterMap<KT,CT>::wv2cv(std::vector<KT> &v) {
  const int order=v.size();
  assert(order<=m_map.size());
  for (int i=1;i<=order;i++) {
    v[order-i]=get_cluster(i,v[order-i]);
  }
}

template <typename KT, typename CT>
void ClusterMap<KT, CT>::wg2cg(TreeGram::Gram &g) {
  const int order=g.size();
  //assert(order<m_map.size());
  for (int i=1;i<=std::min((size_t) order,m_map.size()-1);i++) {
    g[order-i]=get_cluster(i,g[order-i]);
  }
}
