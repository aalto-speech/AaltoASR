#!/usr/bin/perl

# Removes morph boundary symbols from FST stream

while (<>) {
  if (/\#\d+/) { # Word boundary input symbol
    if (!/#\d+\s+(\S+)/ || $1 eq ",") {
      print "Invalid word boundary: $_";
      exit 1;
    }
    s/#\d+\s+(\S+)/#$1/;
  }
  print
}
