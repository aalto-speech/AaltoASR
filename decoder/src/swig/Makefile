includes = -I.. -I.
include ../Makefile.common

SWIG = swig
PYTHONINC = $(shell python path.py)

libs = ../libdecoder.a ../misc/libmisc.a ../fsalm/libfsalm.a

##################################################

default: Decoder.py _Decodermodule.so

Decoder_wrap.cc Decoder.py: Decoder.i $(libs)
	$(SWIG) -I../ -c++ -python -shadow -o Decoder_wrap.cc Decoder.i

_Decodermodule.so: Decoder_wrap.cc $(libs)
	$(CXX) $(cxxflags) -shared -I$(PYTHONINC) -I../ Decoder_wrap.cc $(libs) -o _Decodermodule.so

.PHONY: clean
clean:
	rm -f Decoder_wrap.cc Decoder.py _Decodermodule.so *~
