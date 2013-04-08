includes = -I..
include ../Makefile.common

##################################################

progs = lm

libfsalm_srcs = LM.cc ArpaReader.cc

##################################################

libs = libfsalm.a ../misc/libmisc.a
progs_srcs = $(progs:=.cc)
progs_objs = $(progs:=.o)
libfsalm_objs = $(libfsalm_srcs:.cc=.o)
all_srcs = $(libfsalm_srcs) $(progs_srcs)
all_objs = $(all_srcs:.cc=.o)

##################################################

.SUFFIXES:

default: libfsalm.a progs

.PHONY: progs
progs: libfsalm.a $(progs)

%: %.o $(libs)
	$(CXX) $< -o $@ $(libs)

%.o: %.cc
	$(CXX) -c $(cxxflags) $< -o $@

libfsalm.a: $(libfsalm_objs)
	ar r libfsalm.a $(libfsalm_objs)

.PHONY: clean
clean:
	rm -f $(all_objs) libfsalm.a .depend *~ $(progs)

dep:
	$(CXX) -MM $(cxxflags) $(DEPFLAGS) $(all_srcs) > dep
include dep
