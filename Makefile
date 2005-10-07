OPT = -g
INCLUDES = -I/share/puhe/suse93/include 
CXXFLAGS = $(OPT) $(INCLUDES) -Wall $(shell sdl-config --cflags)
LDFLAGS = -L/share/puhe/suse93/lib -Wl,-rpath -Wl,/share/puhe/suse93/lib
LIBS = -lsndfile $(shell sdl-config --libs)

PLAYSEG_OBJS = \
	playseg.o \
	AudioPlayer.o \
	io.o conf.o str.o

SRCS += $(PLAYSEG_OBJS:.o=.cc)
OBJS = $(PLAYSEG_OBJS)
PROGS = playseg

all: $(PROGS)

%.o: %.cc Makefile
	$(CXX) $(CXXFLAGS) -c $<

playseg: $(PLAYSEG_OBJS)
	$(CXX) $(LDFLAGS) $(CXXFLAGS) -o $@ $(PLAYSEG_OBJS) $(LIBS)

.PHONY: clean
clean:
	rm -f $(PROGS) $(OBJS)

dep:
	$(CXX) -MM $(CPPFLAGS) $(CXXFLAGS) $(SRCS) > dep

include dep