#!/usr/bin/perl

use Hmm;
use Fsm;
use strict 'vars';

my $eps = ",";
my $boundary = "#";
my $final_symbol = "-2";
my $max_boundary_id = 0;

if (open(FILE, "boundaries")) {
  $max_boundary_id = <FILE>;
  chomp $max_boundary_id;
  close(FILE);
}

$Hmm::verbose = 0;
my $hmmfile = shift @ARGV;
my $hmms = Hmm::read($hmmfile);
my @accept = @ARGV;

my %state_map = ();
my @labels = ();
my @arcs = ();

# Generate FSM
my $fsm = Fsm->new;
$fsm->set_initial($fsm->add_node());
my $final = $fsm->add_node();
$fsm->set_final($final);

my @short_silences = ("_");
my @filler_models = ();

for my $label (keys %$hmms) {

    my $accept = 0;
    for (@accept) {
	$accept = 1 if (eval "\$label =~ /$_/");
    }
    next if ($accept == 0);

    $label =~ /^(\S+-)?(\S+?)(\+\S+)?$/;
    my $l = $1;
    my $c = $2;
    my $r = $3;

    my $tgt_ctx = "$c$r";
    $tgt_ctx =~ tr/+/-/;
    $tgt_ctx = "-__" if ($r eq "+_");
    $tgt_ctx = "__-" if ($label eq "__");

    my $state_id = $state_map{$tgt_ctx};
    if ($state_id eq undef) {
	$state_id = $fsm->add_node();
	$state_map{$tgt_ctx} = $state_id;
    }
    push(@labels, $label);
}

for my $label (@labels) {
    next if ($label eq "_");
    next if ($label eq "__");

    if ($label =~ /^_[^\-\+]+$/) {
      # Filler model, can be instatiated between any two silences or
      # similar to short silence
      push(@short_silences, $label);
      push(@filler_models, $label);
      next;
    }

    $label =~ /^(\S+-)?(\S+?)(\+\S+)?$/;
    my $l = $1;
    my $c = $2;
    my $r = $3;

    my $src_ctx = "$l$c";
    $src_ctx = "__-" if ($l eq "_-");

    my $tgt_ctx = "$c$r";
    $tgt_ctx =~ tr/+/-/;
    $tgt_ctx = "-__" if ($r eq "+_");

    my $src = $state_map{$src_ctx};
    my $tgt = $state_map{$tgt_ctx};

    if ($src eq undef) {
	print STDERR "WARNING: source does not exist, skipping: $label\n";
	next;
    }

    if ($tgt eq undef) {
	print STDERR "WARNING: target does not exist, skipping: $label\n";
	next;
    }

    $fsm->add_arc($src, $tgt, $label, $c, 0);
}

$fsm->add_arc($fsm->initial, $state_map{"-__"}, $eps, $eps, 0);
$fsm->add_arc($state_map{"__-"}, $final, $eps, $eps, 0);
$fsm->add_arc($state_map{"-__"}, $state_map{"__-"}, "__", "__", 0);
if (@filler_models) {
  my $filler_source = $fsm->add_node();
  my $filler_target = $fsm->add_node();

  # To fill the morph boundary arcs, add these to the state map
  $state_map{__f} = $filler_source;
  $state_map{_f_} = $filler_target;

  $fsm->add_arc($state_map{"-__"}, $filler_source, "__", "__", 0);
  $fsm->add_arc($state_map{"-__"}, $filler_source, "_", "_", 0);
  $fsm->add_arc($filler_target, $state_map{"__-"}, "__", "__", 0);
  $fsm->add_arc($filler_target, $state_map{"__-"}, "_", "_", 0);

  foreach my $fmlab (@filler_models) {
    $fsm->add_arc($filler_source, $filler_target, $fmlab, $fmlab, 0);
  }
}

while ((my $ctx, my $id) = each %state_map) {
    if ($boundary ne undef) {
	for my $i (1..$max_boundary_id) {
	    $fsm->add_arc($id, $id, "$boundary$i", "$boundary$i", 0);
	}
    }
    
    next if ($ctx !~ /\S+-\S+/);
    foreach my $sslab (@short_silences) {
      $fsm->add_arc($id, $id, $sslab, $sslab, 0);
    }
}

$fsm->print_fst();


