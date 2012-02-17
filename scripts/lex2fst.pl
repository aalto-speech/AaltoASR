#!/usr/bin/perl

my $eps = ",";
my $boundary = "#";
my $final_symbol = $eps; # "-2"
my $sentence_end = "</s>";
my %boundary_hash = ();
my $max_boundary_id = -1;
my $morphs = 0; # If true, does not generate word breaks automatically

use Fsm;
use strict;
use warnings;



# Open input
#

my $fh;
my $lexfile = shift @ARGV;
if ($lexfile eq "-m") {
  $morphs = 1;
  $lexfile = shift @ARGV;
}
$lexfile = "-" if (not defined $lexfile);

if ($lexfile eq "-") {
  $fh = "STDIN";
}
else {
  open($fh, "$lexfile") || die("could not open '$lexfile'");
}


# Generate FST
#

my $fsm = Fsm->new;
$fsm->set_initial($fsm->add_node());

my $wordbreak;
$wordbreak = $fsm->add_node() if (!$morphs);

while (<$fh>) {
    chomp;
    next if (/^\s*$/);

    s/\S+\-(\S+)\+\S+/$1/g; # Reduce context phones to monophones

    my @fields = split(/\s+/);
    my $out = shift(@fields);
    $out =~ s/\([0-9\.]+\)//;

    if (!$morphs && substr($out, 0, 1) eq "_") {
      add_to_fst($wordbreak, $fsm->initial, $eps, @fields);
      if ($out eq "__") {
        # Create also normal word
        add_to_fst($fsm->initial, $fsm->initial, $out, @fields);
      }
    } else {
      if (!$morphs && scalar @fields > 0) {
        add_to_fst($fsm->initial, $wordbreak, $out, @fields);
      } else {
        # Morphs or an empty word
        add_to_fst($fsm->initial, $fsm->initial, $out, @fields);
      }
    }
}

open(FILE, ">boundaries");
print FILE "$max_boundary_id\n";
close(FILE);

# The next line allows skipping the word brake HMMs 
#$fsm->add_arc($wordbreak, $fsm->initial, $eps, $eps, 0);

my $final = $fsm->add_node();
$fsm->set_final($final);
$fsm->add_arc($fsm->initial, $final, $final_symbol, $sentence_end, 0);
$fsm->add_arc($fsm->initial, $final, $final_symbol, $eps, 0);

$fsm->print_fst();

close $fh if ($fh ne "STDIN");


sub add_to_fst {
  my $start_node = shift(@_);
  my $end_node = shift(@_);
  my $label = shift(@_);

  my $hmm_string = join(" ", @_);
  $boundary_hash{$hmm_string}++;
  my $boundary_id = $boundary_hash{$hmm_string};
  $max_boundary_id = $boundary_id if ($boundary_id > $max_boundary_id);
  my $boundary_str = "$boundary$boundary_id";
  my $src = $fsm->add_node();
  my $tgt;
  $fsm->add_arc($start_node, $src, $boundary_str, $label, 0);

  for (@_) {
    $tgt = $fsm->add_node();
    $fsm->add_arc($src, $tgt, $_, $eps, 0);
    $src = $tgt;
  }
  $fsm->add_arc($src, $end_node, $boundary_str, $eps, 0);
}
