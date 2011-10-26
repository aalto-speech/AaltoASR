#!/usr/bin/perl

# Converts transcript from standard input to triphone phn

use locale;
use strict;

while (<>) {
  triphonize($_);
}

sub triphonize {
  my $sent = shift(@_);

  $sent =~ s/^\s+//;
  $sent =~ s/\s+$//;
  $sent =~ s/\s+/ /g;
  $sent =~ s/ _ /_/g;
  
  my (@left_queue, @center_queue, @right_queue);
  my $l;
  my $cur_left_context;

  my @llist = split(//, $sent);

  # Add the initial silence
  $cur_left_context = "_";
  push @left_queue, "_";
  push @center_queue, "__";

  while (defined ($l = shift @llist)) {
    if ($l eq "_") {
      push @center_queue, "__";
    } elsif ($l eq " ") {
      push @center_queue, "_";
    } else {
      push @center_queue, $l;
    }
    # Short silence is a special case, as it does not affect the context
    if ($l ne " ") {
      my $right_context = $l;
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

  # Add the last silence context
  push @right_queue, "_"; # if ($#right_queue+1 < $#center_queue);

  # Add the ending silence
  $cur_left_context = "_";
  push @left_queue, "_";
  push @center_queue, "__";
  push @right_queue, "_";

  # Print the queues
  for (my $i = 0; $i <= $#center_queue; $i++) {
    if (substr($center_queue[$i], 0, 1) eq "_") {
      print $center_queue[$i]."\n";
    } else {
      print $left_queue[$i]."-".$center_queue[$i]."+".$right_queue[$i]."\n";
    }
  }
}
