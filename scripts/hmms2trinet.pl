#!/usr/bin/perl

my $eps = ",";
my $boundary = "#";
my $final_symbol = "-2";

open(FILE, "boundaries");
my $max_boundary_id = <FILE>;
chomp $max_boundary_id;
close(FILE);

use Hmm;
use Fsm;
use strict 'vars';

$Hmm::verbose = 0;
my $hmmfile = shift @ARGV;
my $hmms = Hmm::read($hmmfile);
my @accept = @ARGV;

my %state_map = ();
my @labels = ();
my @arcs = ();

# Generate FSM
my $fsm = Fsm->new;
$fsm->{i} = $fsm->add_node();
my $final = $fsm->add_node();
$fsm->set_final($final);

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

$fsm->add_arc($fsm->{'i'}, $state_map{"-__"}, $eps, $eps, 0);
$fsm->add_arc($state_map{"__-"}, $final, $eps, $eps, 0); #$final_symbol, $final_symbol, 0);
$fsm->add_arc($state_map{"-__"}, $state_map{'__-'}, "__", "__", 0);
$fsm->add_arc($state_map{"__-"}, $state_map{'-__'}, $eps, $eps, 0);

while ((my $ctx, my $id) = each %state_map) {
    if ($boundary ne undef) {
	for my $i (1..$max_boundary_id) {
	    $fsm->add_arc($id, $id, "$boundary$i", "$boundary$i", 0);
	}
    }
    
    next if ($ctx !~ /\S+-\S+/);
    $fsm->add_arc($id, $id, "_", "_", 0);
}

$fsm->print_fst();


