#!/usr/bin/perl

# Convert (triphone) phn to transcript

use locale;
use POSIX;

my $cur_word = "";

while (<>)
{
  my $l = "";
  chomp;
  @c = split;
  if ($#c > 0) {
    if (!($c[2] =~ /[^\.]+\.(\d+)$/) || $1 == 0) {
      if ($c[2] =~ /.\-(.)\+./) {
        $l = $1;
      } else {
        $l = $c[2]; # Monophone
        $l =~ s/\.\d+$//;
      }
    }
  } else {
    $l = $c[0]; # PHN without time information, assumes monophones
  }
  if ($l =~ /_+\./ || $l =~ /_+$/) {
    print $cur_word." " if ($cur_word ne "");
    $cur_word = "";
  } else {
    $cur_word = $cur_word.$l;
  }
}

print "$cur_word\n";
