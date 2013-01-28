// Routines for reading and writing arpa format files from and to the 
// internal prefix tree format.

#include <cstddef>  // NULL
#include <stdlib.h>
#include <cassert>

#include "GramSorter.hh"
#include "TreeGramArpaReader.hh"
#include "misc/str.hh"
#include "ArpaReader.hh"
#include "def.hh"

TreeGramArpaReader::TreeGramArpaReader()
{
}

void
TreeGramArpaReader::read(FILE *file, TreeGram *tree_gram, bool add_missing_unigrams)
{
  std::string line;
  ArpaReader areader(tree_gram);
  bool ok = true;
  bool interpolated;

  areader.read_header(file, interpolated, line);
  if (interpolated) {
    tree_gram->set_type(TreeGram::INTERPOLATED);
  }

  int number_of_nodes = 0;
  for(std::vector<int>::iterator j=areader.counts.begin();j!=areader.counts.end();++j)
    number_of_nodes += *j;

  tree_gram->reserve_nodes(number_of_nodes);
  std::vector<int> tmp_gram;

  float log_prob, back_off;
  int prev_order = 1;
  
  GramSorter *sorter = new GramSorter(1, areader.counts[0]);
  while ( areader.next_gram(file, line, tmp_gram, log_prob, back_off)) {
    int cur_order = tmp_gram.size();
    TreeGram::Gram gram(tmp_gram.begin(), tmp_gram.end());

    if (cur_order > prev_order) {
      // Sort all grams read above and add them to the tree gram.
      sorter->sort();
      assert(sorter->num_grams() == areader.counts[prev_order - 1]);
      for (int i = 0; i < sorter->num_grams(); i++) {
        GramSorter::Data data = sorter->data(i);
        TreeGram::Gram gram = sorter->gram(i);
        tree_gram->add_gram(gram, data.log_prob, data.back_off, add_missing_unigrams);
      }
      delete sorter;
      sorter = new GramSorter(cur_order, areader.counts[cur_order-1]);
      prev_order=cur_order;
    }
    sorter->add_gram(gram, log_prob, back_off);
  }
  // FIXME: Repeating the same code
  // Finally, sort and add the highest order
  assert(sorter->num_grams() == areader.counts.back());
  sorter->sort();
  for (int i = 0; i < sorter->num_grams(); i++) {
    GramSorter::Data data = sorter->data(i);
    TreeGram::Gram gram = sorter->gram(i);
    tree_gram->add_gram(gram, data.log_prob, data.back_off, add_missing_unigrams);
  }
  delete sorter;
  tree_gram->finalize(add_missing_unigrams);
}

void
TreeGramArpaReader::write(FILE *out, TreeGram *tree_gram) 
{
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
      for (int j = 1; j <= order; j++) {
	    fprintf(out, " %s", tree_gram->word(iter.node(j).word).c_str());
      }

      // Possible backoff
      if (iter.has_children())
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
      for (int j = 1; j <= order; j++) {
        fprintf(out, " %s", tree_gram->word(indices[j-1]).c_str());
      }

      // Possible backoff
      if (iter.has_children()) 
        fprintf(out, " %g\n", iter.node().back_off);
      else
        fprintf(out, "\n");
    }
  }
  fprintf(out, "\n\\end\\\n");
}
