PYTHONINC = -I$(PYTHON_INCLUDE_PATH)

all: Decoder.py _Decodermodule.so

Decoder_wrap.cc Decoder.py: Decoder.i ../libdecoder.a
	$(SWIG) -I../ -c++ -python -shadow -o Decoder_wrap.cc Decoder.i

_Decodermodule.so: Decoder_wrap.cc ../libdecoder.a
	$(CXX) $(CXXFLAGS) -shared $(PYTHONINC) -I../ Decoder_wrap.cc ../libdecoder.a -o _Decodermodule.so

EXTRA_DIST = Decoder.i

CLEANFILES = Decoder_wrap.cc Decoder.py _Decodermodule.so