include Makefile.common

##################################################

progs = arpa2bin

libdecoder_srcs = ArpaNgramReader.cc \
	BinNgramReader.cc \
	Endian.cc \
	Expander.cc \
	GramSorter.cc \
	Hmm.cc \
	Lexicon.cc \
	LnaReaderCircular.cc \
	Ngram.cc \
	NowayHmmReader.cc \
	NowayLexiconReader.cc \
	OneFrameAcoustics.cc \
	Search.cc \
	TPLexPrefixTree.cc \
	TPNowayLexReader.cc \
	TokenPassSearch.cc \
	Toolbox.cc \
	TreeGram.cc \
	TreeGramArpaReader.cc \
	Vocabulary.cc \
	io.cc \
	str.cc \
	tools.cc

##################################################

libdecoder_objs = $(libdecoder_srcs:.cc=.o)
progs_srcs = $(progs:=.cc)
all_srcs = $(libdecoder_srcs) $(progs_srcs)
all_objs = $(all_srcs:.cc=.o)

##################################################

default: libdecoder.a $(progs) swig

%: %.o
	$(CXX) $< -o $@ libdecoder.a

%.o: %.cc
	$(CXX) -c $(cxxflags) $< -o $@

libdecoder.a: $(libdecoder_objs)
	ar r libdecoder.a $(libdecoder_objs)

.PHONY: swig
swig:
	$(MAKE) -C swig

.PHONY: clean
clean:
	rm -f $(all_objs) $(progs) libdecoder.a .depend *~
	rm -rf html
	$(MAKE) -C swig clean

dep:
	$(CXX) -MM $(cxxflags) $(DEPFLAGS) $(all_srcs) > dep
include dep
