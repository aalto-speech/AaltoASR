Aalto ASR Tools
===============

Aku
---

Aku is the acoustic modeling toolkit of Aalto University.


Aalto Decoder
-------------

Decoder is a C++ library that reads a language model and a pronunciation dictionary, builds the search tree, and performs the actual speech recognition. It can be linked to a C++ program, or called from Python code through a SWIG wrapper. Decoder requires acoustic probabilities computed using Aku library.

Sample code for how to use Aalto ASR to perform speech recognition in C++ is in `decoder/decode-stream.cc`.


PyRecTool
---------

PyRecTool is a Python tool that performs automatic speech recognition using Aalto ASR.


Installation
============

See INSTALLATION.md


License
=======

All the code in the AaltoASR package is licensed with the Modified BSD license (see also LICENSE). 

Code included from other packages (all in vendors) are licensed according their original project. You can find those licenses in the top of the source files or a LICENSE file in the appropriate directory.
