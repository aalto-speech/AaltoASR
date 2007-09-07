#!/usr/bin/perl

my $boundary = "#";
my $eps = ",";

use Hmm;
use Fsm;
use strict 'vars';

open(FILE, "boundaries");
my $max_boundary_id = <FILE>;
chomp $max_boundary_id;
close(FILE);

$Hmm::verbose = 0;
my $hmmfile = shift @ARGV;
my $hmms = Hmm::read($hmmfile);

my @accept = @ARGV;


# Generate FSM
my $fsm = Fsm->new;
$fsm->set_initial($fsm->add_node());
my $hmm_end = $fsm->add_node();

for my $label (keys %$hmms) {

    my $accept = 0;
    for (@accept) {
	$accept = 1 if (eval "\$label =~ /$_/");
    }
    next if ($accept == 0);

    my $hmm = $hmms->{$label};
    my $hmm_states = $hmm->{'states'};
    my @fsm_states = ();
    for my $state (@$hmm_states) {
	push(@fsm_states, $fsm->add_node());
    }

    for my $s (0..scalar @$hmm_states - 1) {
	my $out = "$eps";
	my $arcs = $hmm_states->[$s]->{'arcs'};
	for my $arc (@$arcs) {
#	    $out = $label if ($arc->[0] == 1);
            my $in = $arc->[2];
            $in = $eps if ($in == -1);
	    $fsm->add_arc($fsm_states[$s], $fsm_states[$arc->[0]], 
			  $in, $out, 0); #log($arc->[1])
	}
    }
    $fsm->add_arc($fsm->initial, $fsm_states[0], "$eps", "$label", 0);
    $fsm->add_arc($fsm_states[1], $hmm_end, "$eps", "$eps", 0);
}

#$fsm->add_arc($hmm_end, $fsm->initial, "$eps", "$eps", 0);

if ($boundary ne undef) {
    for my $i (1..$max_boundary_id) {
	$fsm->add_arc($hmm_end, $hmm_end, "$boundary$i", "$boundary$i", 0);
    }
}

my $final = $fsm->add_node();
$fsm->set_final($final);

#my $final_symbol = $Hmm::greatest_pdf + 1;
my $final_symbol = -2;

$fsm->add_arc($hmm_end, $final, "$eps", "$eps", 0);
open(FILE, ">final.fst") || die("could not open final.fst");
print FILE "#FSTBasic MaxPlus
I 0
F 1
T 0 1 $final_symbol $final_symbol
";

$fsm->print_fst();


