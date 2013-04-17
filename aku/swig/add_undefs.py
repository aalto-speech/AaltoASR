#!python

import sys

for l in sys.stdin:
  sys.stdout.write(l)
  if l=="#include <Python.h>\n":
    sys.stdout.write("#undef c_abs\n")
    #sys.stdout.write("#undef _c_Py_abs\n")

