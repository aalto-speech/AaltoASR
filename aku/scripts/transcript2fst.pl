#!/usr/bin/perl

use strict;

my @c;
my $node = 0;

print "#FSTBasic MaxPlus\nI 0\n";

while (<>) {
  chomp;
  @c = split;
  while ($#c >= 0) {
    my $w = shift @c;
    print "T ".$node." ".($node+1)." $w $w\n";
    $node++;
  }
}
print "F ".$node."\n";
