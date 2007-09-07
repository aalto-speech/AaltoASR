#!/usr/bin/perl

print "#FSTBasic MaxPlus\n";


while (<>) {
    if (/start=(\d+)/) {
      $initial = $1;
      print "I $initial\n";
    }
    if (/end=(\d+)/) {
      $final = $1;
      print "F $final\n";
    }

    if (/^J=/) {
	$s = $1 if (/S=(\d+)/);
	$e = $1 if (/E=(\d+)/);
	$w = $1 if (/W=(\S+)/);
	$l = $1 if (/l=(\S+)/);

	$w =~ s/!NULL/,/g;
	#$a = $a * log(10);
	#$a = -$a if ($a < 0);

        $l = $l * log(10); # Only keep the language model score

	printf("T $s $e $w $w %g\n", $l);
    }
}
print "$final\n";
