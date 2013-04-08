#!/usr/bin/perl

# Sets the alignment and hmmnet fields in the recipe file. These are
# the ones used for output during training, so they should be unique
# to the different model trainings (the hmmnets can be shared if the
# context phoneme tying remains the same).

# The script is given the recipe and path prefixes as arguments. The
# first path prefix is the one already in the transcript-field and
# determines the inner path structure for the output fields. Then
# alignment and hmmnet path prefixes follow.

# Example:
#   set_recipe_out_paths.pl speecon_adult_train.recipe \
#                           /share/puhe/audio/speecon-fi/adult/phn \
#                           /MY_WORKDIR/phn /MY_WORKDIR/hmmnet

use locale;
use strict;

my $recipe = shift;
my $tr_prefix = shift;
my $alignment_prefix = shift;
my $hmmnet_prefix = shift;

my $fh;

die "Usage: set_recipe_out_paths.pl RECIPE transcript-prefix alignment-prefix hmmnet-prefix" if (!defined $hmmnet_prefix);

$tr_prefix.="/" if (substr($tr_prefix, -1) ne "/");
$alignment_prefix.="/" if (substr($alignment_prefix, -1) ne "/");
$hmmnet_prefix.="/" if (substr($hmmnet_prefix, -1) ne "/");

open($fh, "<$recipe") || die "Could not open recipe file $recipe\n";

while (<$fh>) {
  chomp;
  my $recipe_line = $_;
  my $path;
  /transcript=(\S*)/;
  $path = $1;
  $path=~s/$tr_prefix//;
  $path=~s/\.phn//;
  if (length($path) > 0) {
    $recipe_line.=" alignment=".$alignment_prefix.$path.".rawseg hmmnet=".$hmmnet_prefix.$path.".hmmnet\n";
  }
  print $recipe_line;
}
