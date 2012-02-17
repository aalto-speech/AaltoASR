package Fsm;

use strict;
use warnings;

sub new {
    my ($type) = @_;
    my $self = {};

    $self->{"i"} = -1;
    $self->{"F"} = {};
    $self->{"free_state"} = 0;
    $self->{"V"} = [];
    $self->{"eps"} = ",";

    bless $self, $type;
    return $self;
}

sub initial {
    my ($self) = @_;
    return $self->{"i"};
}

sub arcs {
    my ($self, $state) = @_;
    return $self->{"V"}->[$state];
}

sub is_final {
    my ($self, $state) = @_;
    return $self->{"F"}->{$state};
}

sub set_final {
    my ($self, $state) = @_;
    $self->{"F"}->{$state} = 1;
}

sub set_initial {
    my ($self, $state) = @_;
    $self->{"i"} = $state;
}

sub add_node {
    my ($self) = @_;
    my $state = $self->{"free_state"}++;
    $self->{"V"}->[$state] = [];
    return $state;
}

sub add_arc {
    my ($self, $src, $tgt, $in, $out, $weight) = @_;

    $in = $self->{"eps"} if (not defined $in);
    $out = $self->{"eps"} if (not defined $out);
    $weight = 0 if (not defined $weight);

    die("Fsm::addarc(): invalid transition: $src $tgt $in $out $weight\n")
	if (not defined $src || not defined $tgt);

    push(@{$self->{"V"}->[$src]}, [$src, $tgt, $in, $out, 
				   sprintf("%g", $weight)]);
}

sub read_fst {
    my ($self, $filename) = @_;

    my $fh;
    if ($filename eq "-") {
	$fh = "STDIN";
    }
    else {
	open($fh, $filename);
    }
    my $line_no = 1;
    while (<$fh>) {
	if ($line_no == 1) {
	    if (/^\#FSTBinary/) {
		print STDERR "can not read binary format\n";
		exit(1);
	    }
	}
	
	if (/^I\s+(\d+)/) {
	    $self->{"i"} = $1;
	    next;
	}

	if (/^F\s+(\d+)/) {
	    $self->{"F"}->{$1} = 1;
	    next;
	}
	
	next if (!/^T/);

	(my $dummy, my $src, my $tgt, my $in, my $out, my $weight) = 
	    split(/\s+/);
        $self->add_arc($src, $tgt, $in, $out, $weight);
    }
    close $fh if ($filename ne "-");
}

sub print_fst {
    my ($self, $fh) = @_;
    $fh = *STDOUT if (not defined $fh);

    print $fh "#FSTBasic MaxPlus\n";
    print $fh "I $self->{i}\n";
    for (keys %{$self->{"F"}}) {
	print $fh "F $_\n";
    }
    for my $s (@{$self->{"V"}}) {
	for (@$s) {
	    print $fh "T ", join(" ", @$_), "\n";
	}
    }
	
}

sub push_outputs {
    my ($self) = @_;

    # Compute incoming arcs
    #
    my @queue = ();
    my @incoming = ();
    my $states = $self->{"V"};
    for my $src (0..scalar @$states - 1) {
	for my $arc (@{$states->[$src]}) {
	    push(@{$incoming[$arc->[1]]}, $arc);
	}
	push(@queue, $src);
    }

    # Push each label once
    # 
    my $eps = $self->{"eps"};
    while (scalar @queue > 0) {
	my $src = shift @queue;

	my $out_arcs = $states->[$src];
	next if (scalar @$out_arcs != 1);
	my $in_arcs = $incoming[$src];
	next if (scalar @$in_arcs != 1);

	my $out_arc = $out_arcs->[0];
	next if ($out_arc->[3] eq $eps);
	my $in_arc = $in_arcs->[0];
	next if ($in_arc->[3] ne $eps);

	$in_arc->[3] = $out_arc->[3];
	$out_arc->[3] = $eps;
	push(@queue, $in_arc->[0]);
    }
}

1;
