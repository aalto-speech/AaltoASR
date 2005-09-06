OPT = -g
INCLUDES =
CXXFLAGS = $(OPT) $(INCLUDES) -Wall
LDFLAGS = 
LIBS =

morph_lattice_OBJS = \
	morph_lattice.o \
	MorphSet.o \
	io.o conf.o str.o

SRCS += $(morph_lattice_OBJS:.o=.cc)
OBJS = $(morph_lattice_OBJS)
PROGS = morph_lattice

all: $(PROGS)

%.o: %.cc
	$(CXX) $(CXXFLAGS) -c $<

morph_lattice: $(morph_lattice_OBJS)
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $(morph_lattice_OBJS) $(LIBS)

.PHONY: clean
clean:
	rm -f $(PROGS) $(OBJS)

dep:
	$(CXX) -MM $(CPPFLAGS) $(CXXFLAGS) $(SRCS) > dep

include dep