OPT = -O2
INCLUDES = 
CXXFLAGS = $(OPT) $(INCLUDES) -Wall
LDFLAGS =
LIBS = 

LATTICE_RESCORE_OBJS = \
	lattice_rescore.o \
	Endian.o Lattice.o Rescore.o TreeGram.o Vocabulary.o \
	io.o conf.o str.o

SRCS += $(LATTICE_RESCORE_OBJS:.o=.cc)
OBJS = $(LATTICE_RESCORE_OBJS)
PROGS = lattice_rescore

all: $(PROGS)

%.o: %.cc Makefile
	$(CXX) $(CXXFLAGS) -c $<

lattice_rescore: $(LATTICE_RESCORE_OBJS)
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $(LATTICE_RESCORE_OBJS) $(LIBS)

.PHONY: clean
clean:
	rm -f $(PROGS) $(OBJS)

dep:
	$(CXX) -MM $(CPPFLAGS) $(CXXFLAGS) $(SRCS) > dep

include dep