#!/usr/bin/perl

# Fixes word boundary labels and removes unnecessary out labels.
# The resulting FST should be optimized after this operation.

while (<>) {
  if (/(T\s+\d+\s+\d+\s+)#\d+/) { # Word boundary input symbol
    if (!/#\d+\s+(\S+)/) {
      print STDERR "Invalid word boundary: $_";
      exit 1;
    }
    if ($1 eq ",") {
      s/#\d+\s+(\S+)/, ,/;
    } else {
      s/#\d+\s+(\S+)/#$1 ,/;
    }
  } elsif (/T\s+\d+\s+\d+,\s+(\S+)/) { # Epsilon input label
    if ($1 ne ",") {
      s/(T\s+\d+\s+\d+,\s+)\S+/$1 ,/;
    }
  }
  print;
}
