#!/usr/bin/perl

while (<>) {
  chomp;
  @c = split;
  $c[5] = -$c[5] if ($#c >= 5);
  print join(" ", @c). "\n";
}
