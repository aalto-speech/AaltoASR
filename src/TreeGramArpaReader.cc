#include <stdlib.h>
#include <assert.h>

#include "GramSorter.hh"
#include "TreeGramArpaReader.hh"
#include "str.hh"
#include "ClusterMap.hh"

TreeGramArpaReader::TreeGramArpaReader()
  : m_lineno(0)
{
}

void
TreeGramArpaReader::read_error()
{
  fprintf(stderr, "TreeGramArpaReader::read(): error on line %d\n", m_lineno);
  exit(1);
}

void
TreeGramArpaReader::read(FILE *file, TreeGram *tree_gram)
{
  std::string line;
  std::vector<std::string> vec;

  // Just for efficiency
  line.reserve(128); 
  vec.reserve(16);

  bool ok = true;

  m_lineno = 0;

  // Find header
  while (1) {
    ok = str::read_line(&line, file, true);
    m_lineno++;

    if (!ok) {
      fprintf(stderr, "TreeGramArpaReader::read(): "
	      "error on line %d while waiting \\data\\", m_lineno);
      exit(1);
    }

    if (line.substr(0,11) == "\\clustermap") {
      int ord;
      if (sscanf(line.c_str(),"\\clustermap %d",&ord)!=1) assert(false);
      tree_gram->clmap=new ClusterMap<int>;
      m_lineno=tree_gram->clmap->read(file,ord,m_lineno);
      for (int i=0;i<tree_gram->clmap->num_words();i++) {
	tree_gram->add_word(tree_gram->clmap->word(i));
      }
    } 

    if (line.substr(0,12) == "\\fclustermap") {
      int ord;
      if (sscanf(line.c_str(),"\\fclustermap %d",&ord)!=1) assert(false);
      tree_gram->clmap=new ClusterFMap<int>;
      m_lineno=tree_gram->clmap->read(file,ord,m_lineno);
      for (int i=0;i<tree_gram->clmap->num_words();i++) {
	tree_gram->add_word(tree_gram->clmap->word(i));
      }
    } 

    if (line == "\\interpolated")
      tree_gram->set_type(TreeGram::INTERPOLATED);

    if (line == "\\data\\")
      break;
  }

  // Read header
  int order = 1;
  int number_of_nodes = 0;
  int max_order_count = 0;
  while (1) {
    ok = str::read_line(&line, file, true);
    m_lineno++;

    if (!ok) {
      fprintf(stderr, "TreeGramArpaReader::read(): "
	      "error on line %d while reading counts", m_lineno);
      exit(1);
    }
    
    // Header ends in a \-command
    if (line[0] == '\\')
      break;

    // Skip empty lines
    if (line.find_first_not_of(" \t\n") == line.npos)
      continue;

    // All non-empty header lines must be ngram counts
    if (line.substr(0, 6) != "ngram ")
      read_error();
    {
      std::string tmp(line.substr(6));
      str::split(&tmp, "=", false, &vec);
    }
    if (vec.size() != 2)
      read_error();

    int count = atoi(vec[1].c_str());
    if (count > max_order_count)
      max_order_count = count;
    number_of_nodes += count;
    m_counts.push_back(count);
    
    if (atoi(vec[0].c_str()) != order || m_counts.back() < 0)
      read_error();
    order++;
  }

  tree_gram->reserve_nodes(number_of_nodes);

  // Read ngrams order by order
  for (order = 1; order <= m_counts.size(); order++) {

    // We must always have the correct header line at this point
    if (line[0] != '\\') {
      fprintf(stderr, "TreeGramArpaReader::read(): "
	      "\\%d-grams expected on line %d\n", order, m_lineno);
      exit(1);
    }
    str::clean(&line, " \t");
    str::split(&line, "-", false, &vec);

    if (atoi(vec[0].substr(1).c_str()) != order || vec[1] != "grams:") {
      fprintf(stderr, "TreeGramArpaReader::read(): "
	      "unexpected command on line %d: %s\n", m_lineno, line.c_str());
      exit(1);
    }

    // Read the grams of each order into the sorter
    GramSorter sorter(order, m_counts[order - 1]);
    TreeGram::Gram gram;
    gram.resize(order);
    for (int w = 0; w < m_counts[order-1]; w++) {

      // Read and split the line
      if (!str::read_line(&line, file))
	read_error();
      str::clean(&line, " \t\n");
      m_lineno++;

      // Ignore empty lines
      if (line.find_first_not_of(" \t\n") == line.npos) {
	w--;
	continue;
      }

      str::split(&line, " \t", true, &vec);

      // Check the number of columns on the line
      if (vec.size() < order + 1 || vec.size() > order + 2) {
	fprintf(stderr, "TreeGramArpaReader::read(): "
		"%d columns on line %d\n", vec.size(), m_lineno);
	exit(1);
      }
      if (order == m_counts.size() && vec.size() != order + 1)
	fprintf(stderr, "WARNING: %d columns on line %d\n", vec.size(), 
		m_lineno);

      // FIXME: should we deny new words in higher order ngrams?

      // Parse log-probability, back-off weight and word indices
      // FIXME: check the conversion of floats
      float log_prob = strtod(vec[0].c_str(), NULL);
      float back_off = 0;
      if (vec.size() == order + 2)
	back_off = strtod(vec[order + 1].c_str(), NULL);

      // Add the gram to sorter
      //fprintf(stderr,"add gram [");
      for (int i = 0; i < order; i++) {
	if (tree_gram->clmap) gram[i]=atoi(vec[i+1].c_str());
	else
	gram[i] = tree_gram->add_word(vec[i + 1]);
	//fprintf(stderr," %d", gram[i]);
      }
      sorter.add_gram(gram, log_prob, back_off);
      //fprintf(stderr,"] = [%f %f]\n",log_prob,back_off);
    }

    // Sort all grams read above and add them to the tree gram.
    sorter.sort();
    assert(sorter.num_grams() == m_counts[order - 1]);
    for (int i = 0; i < sorter.num_grams(); i++) {
      GramSorter::Data data = sorter.data(i);
      gram = sorter.gram(i);

      tree_gram->add_gram(gram, data.log_prob, data.back_off);
    }

    // Skip empty lines before the next order.
    while (1) {
      if (!str::read_line(&line, file, true)) {
	if (ferror(file))
	  read_error();
	if (feof(file))
	  break;
      }
      m_lineno++;

      if (line.find_first_not_of(" \t\n") != line.npos)
	break;
    }
  }
  tree_gram->finalize();
}

void
TreeGramArpaReader::write(FILE *out, TreeGram *tree_gram) 
{
  if (tree_gram->clmap) {
    tree_gram->clmap->write(out);
    tree_gram->clear_words();
    char ascii_num[10];
    for (int i=0;i<tree_gram->clmap->num_clusters(1);i++) {
      //fprintf(stderr,"loop %d/%d\n",i,m_clustermap.num_clusters(1));
      sprintf(ascii_num,"%d",i);
      tree_gram->add_word(ascii_num);
    }
  }
  
  if (tree_gram->get_type()==TreeGram::INTERPOLATED) {
    write_interpolated(out,tree_gram);
    return;
  }

  TreeGram::Iterator iter;

  // Header containing counts for each order
  fprintf(out, "\\data\\\n");
  for (int i = 1; i <= tree_gram->order(); i++)
    fprintf(out, "ngram %d=%d\n", i, tree_gram->gram_count(i));

  // Ngrams for each order
  for (int order = 1; order <= tree_gram->order(); order++) {
    iter.reset(tree_gram);
    fprintf(out, "\n\\%d-grams:\n",order);
    while (iter.next_order(order)) {
      
      // Log-probability
      fprintf(out, "%g", iter.node().log_prob);

      // Word indices in the ngram
      for (int j = 1; j <= order; j++)
	fprintf(out, " %s", tree_gram->word(iter.node(j).word).c_str());

      // Possible backoff
      if (order != tree_gram->order())
	fprintf(out, " %g\n", iter.node().back_off);
      else
	fprintf(out, "\n");
    }
  }
  fprintf(out, "\n\\end\\\n");
}


void
TreeGramArpaReader::write_interpolated(FILE *out, TreeGram *tree_gram) 
{
  TreeGram::Iterator iter;

  // Header containing counts for each order
  fprintf(out, "\\data\\\n");
  for (int i = 1; i <= tree_gram->order(); i++)
    fprintf(out, "ngram %d=%d\n", i, tree_gram->gram_count(i));

  // Ngrams for each order
  TreeGram::Gram indices;
  for (int order = 1; order <= tree_gram->order(); order++) {
    indices.resize(order);
    iter.reset(tree_gram);
    fprintf(out, "\n\\%d-grams:\n",order);
    while (iter.next_order(order)) {
      for (int j = 1; j <= order; j++) {
	indices[j-1]=iter.node(j).word;
      }
      
      // Log-probability
      float lp=tree_gram->log_prob_i(indices); // This bypasses Clustermap->wg2cg()

      if (lp>0) {
	fprintf(stderr,"warning, n-gram [");
	for (int j=1;j<=order;j++) 
	  fprintf(stderr," %s", tree_gram->word(indices[j-1]).c_str());
	fprintf(stderr,"] had logprob >0 (%e), corrected\n",lp);
	lp=0;
      }
      fprintf(out, "%g", lp);

      // Word indices in the ngram
      for (int j = 1; j <= order; j++)
	fprintf(out, " %s", tree_gram->word(indices[j-1]).c_str());

      // Possible backoff
      float bo=iter.node().back_off;
      if (bo<-0.0000001)
	fprintf(out, " %g\n", bo);
      else
	fprintf(out, "\n");
    }
  }
  fprintf(out, "\n\\end\\\n");
}
