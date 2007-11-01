#!/usr/bin/perl

# Removes morph boundary symbols from FST stream

while (<>) {
  s/\#\d+$/, /g;
  s/\#\d+\s/, /g;
  print;
}
