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
    if (substr($a->[2], 0, 2) eq "#1" &&
        (length($a->[3]) == 0 || $a->[3] eq $eps)) {
      # Silence arcs do not have out labels after the lexicon
      # expansion, fill them now.
      if ($#{$fsm->{"V"}->[$a->[1]]} == 0 &&
          substr($fsm->{"V"}->[$a->[1]]->[0]->[2], 0, 1) eq "_") {
        $a->[3] = "_";
      }
    }
  }
}

$fsm->print_fst();

