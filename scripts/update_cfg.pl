#!/usr/bin/perl
# updates the old .cfg file to the new format
# where the old fft module is split to audiofile and fft modules
# arg 1 = cfg-file

my $done=0;
open(CFGIN, "<$ARGV[0]");
while (<CFGIN>) {
    
    $line=$_;
    if ($line =~ m/module/ && !$done) {
	$line = <CFGIN>;
	unless ($line =~ m/\{/) {
	    print "ERROR\n";
	}

	my @fft_lines;

	# Divide FFT module
	$line = <CFGIN>;
	if ($line =~ m/name fft/) {
	    $line = <CFGIN>;
	    unless ($line =~ m/type fft/) {
		print "ERROR\n";
	    }
	    
	    while ($line = <CFGIN>) {
		if ($line =~ m/\}/) {
		    last;
		}
		else {
		    push(@fft_lines, $line);
		}
	    }
	}

	print "module\n";
	print "{\n";
	print "  name audiofile\n";
	print "  type audiofile\n";
	for ($i=0; $i<=$#fft_lines; $i++) {
	    unless ($fft_lines[$i] =~ m/magnitude/) {
		print $fft_lines[$i];
	    }
	}
	print "}\n";
	print "\n";
	print "module\n";
	print "{\n";
	print "  name fft\n";
	print "  type fft\n";
	for ($i=0; $i<=$#fft_lines; $i++) {
	    if ($fft_lines[$i] =~ m/magnitude/) {
		print $fft_lines[$i];
	    }
	}
	print "  sources audiofile\n";
	$done=1;
    }

    print $line;
}
