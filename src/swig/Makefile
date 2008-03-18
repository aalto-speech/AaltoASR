include ../Makefile.common

SWIG = swig
PYTHONINC = $(shell python path.py)

##################################################

default: Decoder.py _Decodermodule.so

Decoder_wrap.cc Decoder.py: Decoder.i ../libdecoder.a
	$(SWIG) -I../ -c++ -python -shadow -o Decoder_wrap.cc Decoder.i

_Decodermodule.so: Decoder_wrap.cc ../libdecoder.a
	$(CXX) $(cxxflags) -shared -I$(PYTHONINC) -I../ Decoder_wrap.cc ../libdecoder.a -o _Decodermodule.so

.PHONY: clean
clean:
	rm -f Decoder_wrap.cc Decoder.py _Decodermodule.so *~
