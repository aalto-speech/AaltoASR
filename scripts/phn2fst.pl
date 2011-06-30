#!/usr/bin/perl
# Converts one PHN file to FST, in a format similar to what
# create_hmmnets.pl generates.
# Input: PH-file PHN-file

use Hmm;
use Fsm;
use strict;


my $eps = ",";

$Hmm::verbose = 0;

my $hmmfile = shift @ARGV;
my $hmms = Hmm::read($hmmfile);

my $first = 1;

my $fsm = Fsm->new;

$fsm->set_initial($fsm->add_node());
my $last_state = $fsm->initial;

while (<>) {
  chomp;
  my @c = split;
  if ($#c >= 2) {
    my $full_label = $c[2];
    my $state = 0;
    # Get state number
    if ($full_label=~/\S+\.(\d+)/) {
      $state = $1;
      $full_label =~ s/(\S+)\.\d+/$1/;
    }
    my $pho_label = $full_label;
    if ($pho_label=~/\S+\-(\S+)\+\S+/) {
      $pho_label = $1;
    }
    if ($state == 0) {
      my $hmm = $hmms->{$full_label};
      my $hmm_states = $hmm->{'states'};
      my @fsm_states = ();
      for my $s (1..scalar @$hmm_states - 1) {
        push(@fsm_states, $fsm->add_node());
      }
      $fsm->add_arc($last_state, $fsm_states[1], "#".$pho_label, $eps, 0);
      
      for my $s (2..scalar @$hmm_states - 1) {
        my $out = "$eps";
        my $arcs = $hmm_states->[$s]->{'arcs'};
        for my $arc (@$arcs) {
          my $in;
          if ($arc->[2] >= 0) {
            $in = $arc->[2].";".($s-2).";".$full_label;
            $out = $pho_label;
            # NOTE: Transition probabilities are added on the fly
            $fsm->add_arc($fsm_states[$s-1], $fsm_states[$arc->[0]-1], 
                          $in, $out, 0); #log($arc->[1])
          }
        }
      }
      $last_state = $fsm->add_node();
      $fsm->add_arc($fsm_states[0], $last_state, "#".$pho_label, $eps, 0);
    }
  }
}

# Add end-mark
my $final_node = $fsm->add_node();
$fsm->add_arc($last_state, $final_node, "##E", $eps, 0);
$fsm->set_final($final_node);

$fsm->print_fst();

