#!/usr/bin/perl

# Parses the segmentation output of the decoder and saves segmentations
# of different files to target directory in PHN format. New file
# names are taken from 'LNA: ' line of the decoder output and use
# .snseg suffix.

use locale;


die "Arguments: segmentation-file [target-directory]" if ($#ARGV < 0);

open(SEGF, "< $ARGV[0]");

$dir_prefix = "";
if ($#ARGV > 0) {
  $dir_prefix = $ARGV[1];
  if (substr($dir_prefix, length($dir_prefix)-1, 1) ne '/') {
    $dir_prefix = $dir_prefix."/";
  }
}

$new_segfile = "";

while (<SEGF>) {
  chomp;
  @c = split;
  if ($c[0] eq "LNA:") {
    $new_segfile = $c[1];
    $new_segfile =~ s/\.lna/\.snseg/;
    $new_segfile = $dir_prefix.$new_segfile;
  } elsif ($c[0] eq "REC:") {
    open(OUTF, "> $new_segfile") || die "Can't open $new_segfile\n";
    #print OUTF "".($c[1]*128)." ".($c[2]*128)." ".$c[3]."\n";
    while (<SEGF>) {
      chomp;
      @c = split;
      last if ($c[0] eq "DUR:");
      print OUTF "".($c[0]*128)." ".($c[1]*128)." ".$c[2]."\n";
    }
    close OUTF;
  }
}

close SEGF;
