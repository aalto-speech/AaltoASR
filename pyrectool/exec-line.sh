#!/bin/sh
#
# Executes the command at given line of given a file. First argument is the file
# name and the second argument is the line number starting from 0.
#

file="$1"
line=$(expr $2 + 1)
eval $(sed -n ${line}p "${file}")
