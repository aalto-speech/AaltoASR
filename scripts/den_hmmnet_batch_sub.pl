#!/usr/bin/perl

# DO NOT RUN DIRECTLY, USE generate_den_hmmnets.pl INSTEAD!

use locale;
use strict;

my $BATCH_ID = shift @ARGV;
my $NUM_BATCHES = shift @ARGV;
my $NUM_BLOCKS = shift @ARGV;
my $HMMMODEL = shift @ARGV;
my $LEXICON = shift @ARGV;
my $LMMODEL = shift @ARGV;
my $VOCABULARY = shift @ARGV;
my $RECIPE = shift @ARGV;
my $AKUBINDIR = shift @ARGV;
my $SCRIPTDIR = shift @ARGV;
my $LATTICERECSCRIPT = shift @ARGV;
my $LATTICE_THRESHOLD = shift @ARGV;
my $LMSCALE = shift @ARGV;

# Create own working directory
my $tempdir = "den_sub_temp_${BATCH_ID}";
mkdir $tempdir;

my $num_virtual_batches = $NUM_BATCHES*$NUM_BLOCKS;

for (my $i = 1; $i <= $NUM_BLOCKS; $i++) {
  my $cur_batch = ($BATCH_ID-1)*$NUM_BLOCKS + $i;

  print "Generate LNAs\n";
  # Generate LNAs
  system("$AKUBINDIR/phone_probs -b $HMMMODEL -c $HMMMODEL.cfg -r $RECIPE -o $tempdir -B $num_virtual_batches -I $cur_batch -i 1") && die "phone_probs failed\n";

  print "Generate lattices\n";
  # Generate lattices
  system("$LATTICERECSCRIPT $HMMMODEL $LEXICON $RECIPE $tempdir $num_virtual_batches $cur_batch") && die "recognition failed\n";

  # Remove LNA files
  system("rm $tempdir/*.lna");

  # Generate hmmnets
  system("$SCRIPTDIR/make_den_fst.pl $VOCABULARY $LMMODEL $RECIPE $tempdir $SCRIPTDIR $num_virtual_batches $cur_batch $LATTICE_THRESHOLD $LMSCALE") && die "hmmnet generation failed\n";
}
