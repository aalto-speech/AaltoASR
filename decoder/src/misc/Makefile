include ../Makefile.common

##################################################

libmisc_srcs = Endian.cc io.cc tools.cc conf.cc

##################################################

libmisc_objs = $(libmisc_srcs:.cc=.o)
all_srcs = $(libmisc_srcs)
all_objs = $(all_srcs:.cc=.o)

##################################################

default: libmisc.a

%.o: %.cc
	$(CXX) -c $(cxxflags) $< -o $@

libmisc.a: $(libmisc_objs)
	ar r libmisc.a $(libmisc_objs)

.PHONY: clean
clean:
	rm -f $(all_objs) libmisc.a .depend *~

dep:
	$(CXX) -MM $(cxxflags) $(DEPFLAGS) $(all_srcs) > dep
include dep
