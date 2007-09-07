#!/usr/bin/perl

while (<>) {
    chomp;

    if (/^\d+$/) {
	if ($final ne undef) {
	    print STDERR "ERROR: multiple final states\n";
	    exit 1;
	}
	$final = $&;
	next;
    }

    if (!/^(\d+)\s+(\d+)\s+(\S+)\s*$/) {
	print STDERR "ERROR: weird line: $_\n";
	exit 1;
    }

    $src = $1;
    $tgt = $2;
    $sym = $3;

    $states{$src} = 1;
    $states{$tgt} = 1;

    push(@arcs, [$1, $2, $3]);
}

if ($final eq undef) {
    print STDERR "ERROR: no final state\n";
    exit 1;
}

print "VERSION=1.1\nbase=10\ndir=f\n";
printf "N=%d L=%d\n", scalar keys %states, scalar @arcs;
print "start=0 end=$final\n";

for (keys %states) {
    print "I=$_\n";
}

$j = 0;
for (@arcs) {
    print "J=$j S=$_->[0] E=$_->[1] W=$_->[2]\n";
    $j++;
}
