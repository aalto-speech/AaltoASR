#!/usr/bin/perl

# Removes morph boundary symbols from FST stream

while (<>) {
  s/\#\d+/,/g;
  print;
}
