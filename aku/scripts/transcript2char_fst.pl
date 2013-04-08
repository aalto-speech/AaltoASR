#!/usr/bin/perl

use strict;

my @c;
my $node = 0;

print "#FSTBasic MaxPlus\nI 0\n";

while (<>) {
  chomp;
  s/^\s*//;
  s/\s*$//;
  @c = split(//);
  while ($#c >= 0) {
    my $w = shift @c;
    $w = "<w>" if ($w eq " ");
    print "T ".$node." ".($node+1)." ".$w." ".$w." 0\n";
    $node++;
  }
}
print "F ".$node."\n";
