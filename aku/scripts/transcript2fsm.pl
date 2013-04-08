#!/usr/bin/perl

use strict;

my @c;
my $node = 0;

while (<>) {
  chomp;
  @c = split;
  while ($#c >= 0) {
    my $w = shift @c;
    print $node." ".($node+1)." ".$w."\n";
    $node++;
  }
}
print $node."\n";
