SWIG = swig
PYTHONINC = -I$(PYTHON_INCLUDE_PATH)

all: Decoder.py Decodercmodule.so

Decoder_wrap.o Decoder.py: Decoder.i ../libdecoder.a
	$(SWIG) -I../ -c++ -python -shadow -o Decoder_wrap.cc Decoder.i

Decodercmodule.so: Decoder_wrap.cc ../libdecoder.a
	$(CXX) -shared $(PYTHONINC) -I../ Decoder_wrap.cc ../libdecoder.a -o Decodercmodule.so
