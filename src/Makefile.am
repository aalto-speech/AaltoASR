SUBDIRS = .

noinst_PROGRAMS = test_search test_expander # test_lexicon 

#test_lexicon_SOURCES = test_lexicon.cc
#test_lexicon_LDADD = libdecoder.a

test_expander_SOURCES = test_expander.cc
test_expander_LDADD = libdecoder.a

test_search_SOURCES = test_search.cc
test_search_LDADD = libdecoder.a

noinst_LIBRARIES = libdecoder.a

libdecoder_a_SOURCES = \
					Acoustics.hh \
	Expander.cc			Expander.hh \
	Hmm.cc				Hmm.hh \
	Lexicon.cc			Lexicon.hh \
	LnaReaderCircular.cc		LnaReaderCircular.hh \
	NowayLexiconReader.cc		NowayLexiconReader.hh \
	NowayHmmReader.cc		NowayHmmReader.hh \
	Search.cc			Search.hh \
					Timer.hh \
	Vocabulary.cc			Vocabulary.hh
