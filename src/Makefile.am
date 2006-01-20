SUBDIRS = . swig

noinst_PROGRAMS = arpasort perplexity test_arpa lna arpa2bin test_token_pass
# test_expander 
# test_lexicon 
#	test_arpa

test_token_pass_SOURCES = test_token_pass.cc
test_token_pass_LDADD = libdecoder.a

test_arpa_SOURCES = test_arpa.cc
test_arpa_LDADD = libdecoder.a

arpasort_SOURCES = arpasort.cc
arpasort_LDADD = libdecoder.a

perplexity_SOURCES = perplexity.cc
perplexity_LDADD = libdecoder.a

#test_lexicon_SOURCES = test_lexicon.cc
#test_lexicon_LDADD = libdecoder.a

# test_expander_SOURCES = test_expander.cc
# test_expander_LDADD = libdecoder.a

lna_SOURCES = lna.cc
lna_LDADD = libdecoder.a

arpa2bin_SOURCES = arpa2bin.cc
arpa2bin_LDADD = libdecoder.a

#test_search_SOURCES = test_search.cc
#test_search_LDADD = libdecoder.a

noinst_LIBRARIES = libdecoder.a

libdecoder_a_SOURCES = \
					Acoustics.hh \
	ArpaNgramReader.cc		ArpaNgramReader.hh \
	BinNgramReader.cc		BinNgramReader.hh \
	Endian.cc			Endian.hh \
	Expander.cc			Expander.hh \
	GramSorter.cc			GramSorter.hh \
	Hmm.cc				Hmm.hh \
	Lexicon.cc			Lexicon.hh \
	LnaReaderCircular.cc		LnaReaderCircular.hh \
	Ngram.cc			Ngram.hh \
	NowayLexiconReader.cc		NowayLexiconReader.hh \
	NowayHmmReader.cc		NowayHmmReader.hh \
	Search.cc			Search.hh \
					Timer.hh \
	Toolbox.cc			Toolbox.hh \
	TreeGram.cc			TreeGram.hh \
	TreeGramArpaReader.cc		TreeGramArpaReader.hh \
	tools.cc			tools.hh \
	Vocabulary.cc			Vocabulary.hh \
	TPNowayLexReader.cc		TPNowayLexReader.hh \
	TPLexPrefixTree.cc		TPLexPrefixTree.hh \
	TokenPassSearch.cc		TokenPassSearch.hh \
					Clustermap.hh \
					Clustermap_tmpl.hh \
	io.cc				io.hh \
	str.cc				str.hh