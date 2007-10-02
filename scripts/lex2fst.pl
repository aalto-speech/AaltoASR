#!/usr/bin/perl

my $eps = ",";
my $boundary = "#";
my $final_symbol = $eps; # "-2"
my $sentence_end = "</s>";
my %boundary_hash = ();
my $max_boundary_id = -1;

use Fsm;
use strict;



# Open input
#

my $fh;
my $lexfile = shift @ARGV;
$lexfile = "-" if ($lexfile eq undef);

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

while (<$fh>) {
    chomp;
    next if (/^\s*$/);

    s/\S+\-(\S+)\+\S+/$1/g;

    split(/\s+/);
    my $morph = shift(@_);
    $morph =~ s/\([0-9\.]+\)//;
    $morph = "<w>" if ($morph eq "_"); # Convert short silence to <w>

    my $src = $fsm->initial;
    my $tgt;
    my $hmm_string = join(" ", @_);
    $boundary_hash{$hmm_string}++;
    my $boundary_id = $boundary_hash{$hmm_string};
    $max_boundary_id = $boundary_id if ($boundary_id > $max_boundary_id);

    for (@_) {
	$tgt = $fsm->add_node();
	$fsm->add_arc($src, $tgt, $_, $morph, 0);
	$src = $tgt;
	$morph = "$eps";
    }
    $fsm->add_arc($src, $fsm->initial, "$boundary$boundary_id", $eps, 0);
}

open(FILE, ">boundaries");
print FILE "$max_boundary_id\n";
close(FILE);

my $final = $fsm->add_node();
$fsm->set_final($final);
$fsm->add_arc($fsm->initial, $final, $final_symbol, $sentence_end, 0);
$fsm->add_arc($fsm->initial, $final, $final_symbol, $eps, 0);

$fsm->print_fst();

close $fh if ($fh ne "STDIN");
