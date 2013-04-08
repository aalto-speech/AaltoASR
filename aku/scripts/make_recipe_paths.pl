#!/usr/bin/perl

# Creates all paths needed for one type of files in training
# Input: recipe-file recipe-key
# Example: To create all paths needed for alignment files:
#          ./make_recipe_paths.pl train.recipe alignment


use locale;
use strict;
use File::Path;

my $recipe = shift;
my $key = shift;
my $fh;

die "Usage make_recipe_paths.pl RECIPE key" if (!defined $key);

open($fh, "<$recipe") || die "Could not open recipe file $recipe\n";

while (<$fh>) {
  my $file;
  my $path;
  /$key=(\S*)/;
  $file = $1;
  $file=~/(\S+)\/[^\/]*/;
  $path = $1;
  if (length($path) > 0) {
    if (!(-e $path)) {
      mkpath($path);
    }
  }
}
