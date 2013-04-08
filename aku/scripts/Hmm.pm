package Hmm;

use strict 'vars';
use warnings;

our $verbose = 1;
our $greatest_pdf = -1;

sub read_hmm {
    my $lines = shift;
    my $dummy;

    # Parse HMM specification: HMM STATES LABEL
    $_ = shift @$lines;
    my @fields = split(/\s+/);

    die("invalid HMM specification: $_") if (scalar @fields != 3);
    ($dummy, my $num_states, my $label) = @fields;

    # Parse states
    $_ = shift @$lines;
    @fields = split(/\s+/);
    die("invalid number of states in HMM '$label': $_") 
	if (scalar @fields != $num_states);
    my $states = [];
    for (@fields) {
	$greatest_pdf = $_ if ($_ > $greatest_pdf);
	push(@$states, {"pdf" => $_, "arcs" => []});
    }

    # Parse transitions
    for my $s (0..$num_states - 1) {
	$_ = shift @$lines;
	@fields = split(/\s+/);
	my $bad = 0;
	$bad = 1 if ($fields[0] != $s);
	my $num_arcs = $fields[1];
	$bad = 1 if ($num_arcs < 0);
	$bad = 1 if (($num_arcs * 2 + 2) != scalar @fields);

	die("invalid transition specification in HMM '$label': $_")
	    if ($bad);

        if ($states->[$s]->{"pdf"} >= 0) {
          die("Assuming two arcs per state") if ($num_arcs != 2);

          for my $t (1..$num_arcs) {
	    push(@{$states->[$s]->{"arcs"}}, [$fields[$t * 2], $fields[$t * 2 + 1], $states->[$s]->{"pdf"}*2+$t-1]);
          }
        } else {
          for my $t (1..$num_arcs) {
	    push(@{$states->[$s]->{"arcs"}}, [$fields[$t * 2], $fields[$t * 2 + 1], -1]);
          }
        }
    }

    print STDERR "Hmm: $label  States: $num_states\n" if ($verbose > 0);

    return {"label" => $label, "states" => $states};
}

sub read {
    my $filename = shift;

    # Read all lines
    open(FILE, $filename) || die("package Hmm: could not open $filename\n");
    my @lines = <FILE>;
    chomp @lines;
    close(FILE);

    # Parse header
    $_ = shift @lines;
    die("invalid header $_\n") if ($_ ne "PHONE");
    my $num_hmms = shift @lines;
    print STDERR "Number of HMMs: $num_hmms\n" if ($verbose > 0);

    # Read hmms
    my $hmms = {};
    keys %$hmms = $num_hmms;
    for my $h (1..$num_hmms) {
	my $hmm = read_hmm(\@lines);
	$hmms->{$hmm->{"label"}} = $hmm;
    }

    return $hmms;
}

1;
