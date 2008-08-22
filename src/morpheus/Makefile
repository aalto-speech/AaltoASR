includes = -I..
include ../Makefile.common

##################################################

progs = morpheus

##################################################

libs = ../fsalm/libfsalm.a ../misc/libmisc.a
progs_srcs = $(progs:=.cc)
progs_objs = $(progs:=.o)
all_srcs = $(progs_srcs)
all_objs = $(all_srcs:.cc=.o)

##################################################

.SUFFIXES:

default: progs

.PHONY: progs
progs: $(progs)

%: %.o $(libs)
	$(CXX) $< -o $@ $(libs)

%.o: %.cc
	$(CXX) -c $(cxxflags) $< -o $@

.PHONY: clean
clean:
	rm -f $(all_objs) .depend *~

dep:
	$(CXX) -MM $(cxxflags) $(DEPFLAGS) $(all_srcs) > dep
include dep
