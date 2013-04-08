#!/usr/bin/perl

open(A, $ARGV[0])
    or die("could not open");
open(B, $ARGV[1])
    or die("could not open");

@a = <A>;
@b = <B>;

close(A);
close(B);

die("length mismatch") if ($#a != $#b);

$max = 0;

for $i (0..$#a) {
    $diff = abs($a[i] - $b[i]);
    if ($diff > $max) {
	$max = $diff;
    }
}

print "$#a lines\n";
print "max diff: $max\n";
