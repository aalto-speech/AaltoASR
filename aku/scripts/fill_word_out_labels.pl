#!/usr/bin/perl

my $eps = ",";
my @num_incoming = ();
my @incoming_label = ();

use Fsm;
use strict;


# Load FST
my $fsm = Fsm->new;
$fsm->read_fst(shift);

# Find the word arcs and count incoming arcs
my @word_arcs = ();
for my $s (@{$fsm->{"V"}}) { # Iterate over each state
  for my $a (@$s) { # Iterate over each arc from that state
    my $label = $a->[3]; # Output symbol

    # Don't count self-transitions
    $num_incoming[$a->[1]]++ if ($a->[0] != $a->[1]);

    if (length($label) > 0 && $label ne $eps) {
      if (length($a->[2]) == 0 || $a->[2] eq $eps) {
        # Remove out labels from epsilon arcs
        $a->[3] = $eps;
      } else {
        push(@word_arcs, [$a->[1], $label]);
      }
    }
  }
}

# Fill the word labels
for my $a (@word_arcs) {
  fill_out_labels($a->[0], $a->[1]);
}


$fsm->print_fst();


sub fill_out_labels {
  my $node = shift;
  my $label = shift;

  # Only fill the output labels if they are unambiguous
  return if (!defined $fsm->{"V"}->[$node]);

  if ($num_incoming[$node] > 1) {
    # Only proceed if multiple incoming arcs contain the same out label
    if (!defined $incoming_label[$node]) {
      $incoming_label[$node] = $label;
      return;
    } else {
      return if ($label ne $incoming_label[$node]);
      $num_incoming[$node]--;
      return if ($num_incoming[$node] > 1);
    }
  }

  for my $a (@{$fsm->{"V"}->[$node]}) {
    if (length($a->[3]) == 0 || $a->[3] eq $eps) {
      # Fill output labels only for non-epsilon arcs
      $a->[3] = $label if (length($a->[2]) > 0 && $a->[2] ne $eps);
      
      # Skip self-transitions and do not proceed over boundary markers
      if ($a->[0] != $a->[1] && substr($a->[2], 0, 1) ne "#") {
        fill_out_labels($a->[1], $label);
      }
    }
  }
}
