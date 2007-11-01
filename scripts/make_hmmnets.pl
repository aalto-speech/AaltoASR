#!/usr/bin/perl

# Converts transcription phn files to hmmnet files
# Input: PH-file Recipe
# Reads the transcript-files specified in the recipe
# Writes to hmmnet-files specified in the recipe


use Hmm;
use Fsm;
use strict;
use BSD::Resource;

my $eps = ",";

$Hmm::verbose = 0;

my $hmmfile = shift @ARGV;
my $hmms = Hmm::read($hmmfile);

$SIG{PIPE} = 'IGNORE';
setrlimit(RLIMIT_CORE, 0, 0); # Repress core dumps

my $phn_file;
my $hmmnet_file;
my $fh;
my $retry;

while (<>) {
  chomp;
  if (!/transcript=(\S*)/) {
    print "Missing transcript field, skipping...\n";
    next;
  }
  $phn_file = $1;
  if (!/hmmnet=(\S*)/) {
    print "Missing hmmnet field, skipping...\n";
    next;
  }
  $hmmnet_file = $1;
  for ($retry = 0; $retry < 4; $retry++) {
    open $fh, "| fst_optimize -A - $hmmnet_file" or die "can't run fst_optimize: $!";
    phn2fsm($phn_file, $fh);
    last if (close($fh) && ($?>>8) == 0);
  }
}

sub phn2fsm {
  my ($phn_file, $out_fh) = @_;
  my $first = 1;
  my $last_state = -1;
  my $in_fh;

  my $fsm = Fsm->new;

  open $in_fh, "< $phn_file";

  while (<$in_fh>) {
    chomp;
    my @c = split;
    if ($#c >= 2) {
      my $label = $c[2];
      my $state = 0;
      # Ignore state number
      if ($label=~/\S+\.(\d+)/) {
        $state = $1;
        $label =~ s/(\S+)\.\d+/$1/;
      }
      if ($state == 0) {
        my $hmm = $hmms->{$label};
        my $hmm_states = $hmm->{'states'};
        my @fsm_states = ();
        for my $state (@$hmm_states) {
          push(@fsm_states, $fsm->add_node());
        }
        if ($first) {
          $fsm->set_initial($fsm_states[0]);
          $first = 0;
        }
        if ($last_state >= 0) {
          $fsm->add_arc($last_state, $fsm_states[0], $eps, $eps, 0);
        }

        for my $s (0..scalar @$hmm_states - 1) {
          my $out = "$eps";
          my $arcs = $hmm_states->[$s]->{'arcs'};
          for my $arc (@$arcs) {
            my $in;
            if ($arc->[2] < 0) {
              $in = "#".$label;
            } else {
              $in = $arc->[2]."-".$label.".".($s-2);
            }
            $out = $eps; #$label if ($arc->[0] == 1);
            # NOTE: Transition probabilities are added on the fly
            $fsm->add_arc($fsm_states[$s], $fsm_states[$arc->[0]], 
                          $in, $out, 0); #log($arc->[1])
          }
        }
        $last_state = $fsm_states[1]; 
      }
    }
  }

  close($in_fh);

  $fsm->set_final($last_state);

  $fsm->print_fst($out_fh);
}
