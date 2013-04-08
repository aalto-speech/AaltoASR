#!/usr/bin/perl

# Reads monophone transcripts (with or without time and state information)
# from transcript fields of the recipe file and rewrites them as triphonized 
# versions.

# Input: RECIPE

use locale;
use strict;

my $recipefile = shift;

die "Usage: triphonize_monophone_phns.pl RECIPE" if (!defined($recipefile));

my @files;
@files = load_recipe($recipefile);

for my $f (@files) {
  triphonize($f);
}


sub load_recipe {
  my $recipe_file = shift(@_);
  my @result;
  
  open REC, "< $recipe_file" || die "Could not open $recipe_file\n";
  my @recipe_lines = <REC>;
  close REC;
  my $cur_line = 1;

  foreach my $line (@recipe_lines) {
    $line =~ /transcript=(\S*)/;
    my $trfile = $1;

    die "Error at recipe line ".$cur_line."\n" if ($trfile eq "");

    push(@result, $trfile);
    $cur_line++;
  }
  return @result;
}


sub triphonize {
  my $filename = shift(@_);
  
  my (@time_queue, @left_queue, @center_queue, @right_queue, @state_queue);
  my (@phnf, @t);
  my $cur_left_context;

  my ($in_fh, $out_fh);
  my $phi; # Phoneme index after split, 0 if no time information, 2 otherwise
  my $states; # 0 if no state numbers, 1 if states are present

  $phi = -1;
  $states = -1;
  open $in_fh, "< $filename";
  $cur_left_context = "_";
  while (<$in_fh>) {
    chomp;
    @phnf = split;
    if ($#phnf >= 0) {
      if ($phi == -1) {
        # Determine whether we have time information or not
        if ($#phnf > 1 && $phnf[0] =~ /\d+/ && $phnf[1] =~ /\d+/) {
          $phi = 2;
        } else {
          $phi = 0;
        }
      }
      if ($phi > 0) {
        die "Invalid phn line (file $filename)" if ($#phnf < 2);
        push @time_queue, join(" ", @phnf[0..1]);
      }
      @t = split(/\./, $phnf[$phi]);
      if ($states == -1) {
        $states = 1;
        $states = 0 if ($#t < 1);
      }
      die "Invalid phn line (file $filename)" if ($#t != $states);
      push @center_queue, $t[0];
      push @state_queue, $t[1] if ($states == 1);
      # Short silence is a special case, as it does not affect the context
      if ($#center_queue > 0 && 
          (($states == 1 && $t[1] == 0 && $phnf[$phi] ne "_.0") ||
           ($states == 0 && $phnf[$phi] ne "_"))) {
        my $right_context = $t[0];
        $right_context  = "_" if (substr($right_context, 0, 1) eq "_");
        for (my $i = $#right_queue+1; $i < $#center_queue; $i++) {
          push @right_queue, $right_context;
        }
        for (my $i = $#center_queue-1; $i >= 0; $i--) {
          $cur_left_context = $center_queue[$i];
          last if ($cur_left_context ne "_");
        }
        $cur_left_context = "_" if (substr($cur_left_context, 0, 1) eq "_");
      }
      push @left_queue, $cur_left_context;
    }
  }

  # Fill the right contexts with silence
  for (my $i = $#right_queue+1; $i < $#center_queue; $i++) {
    push @right_queue, "_";
  }

  close $in_fh;

  open $out_fh, "> $filename";

  # Print the queues
  for (my $i = 0; $i <= $#center_queue; $i++) {
    if (substr($center_queue[$i], 0, 1) eq "_") {
      print $out_fh $time_queue[$i]." " if ($phi > 0);
      print $out_fh $center_queue[$i];
      print $out_fh ".".$state_queue[$i] if ($states == 1);
      print $out_fh "\n";
    } else {
      print $out_fh $time_queue[$i]." " if ($phi > 0);
      print $out_fh $left_queue[$i]."-".$center_queue[$i]."+".$right_queue[$i];
      print $out_fh ".".$state_queue[$i] if ($states == 1);
      print $out_fh "\n";
    }
  }

  close $out_fh;
}
