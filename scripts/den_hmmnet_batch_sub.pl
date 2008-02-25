#!/usr/bin/perl

# DO NOT RUN DIRECTLY, USE generate_den_hmmnets.pl INSTEAD!

use locale;
use strict;

my $BATCH_ID = shift @ARGV;
my $NUM_BATCHES = shift @ARGV;
my $NUM_BLOCKS = shift @ARGV;
my $USE_MORPHS = shift @ARGV;
my $HMMMODEL = shift @ARGV;
my $LEXICON = shift @ARGV;
my $LMMODEL = shift @ARGV;
my $VOCABULARY = shift @ARGV;
my $RECIPE = shift @ARGV;
my $AKUBINDIR = shift @ARGV;
my $SCRIPTDIR = shift @ARGV;
my $LNA_OPTIONS = shift @ARGV;
my $LATTICERECSCRIPT = shift @ARGV;
my $LATTICE_THRESHOLD = shift @ARGV;
my $LMSCALE = shift @ARGV;

# Create own working directory
my $tempdir = "den_sub_temp_${BATCH_ID}";
mkdir $tempdir;

my $num_virtual_batches = $NUM_BATCHES*$NUM_BLOCKS;

for (my $i = 1; $i <= $NUM_BLOCKS; $i++) {
  my $cur_batch = ($BATCH_ID-1)*$NUM_BLOCKS + $i;
  my $wg_generated = 0;

  # Load recipe to check if the files already exist (NOTE! only checks
  # the first and the last files in the batch!)
  my $wgs = load_recipe($RECIPE, $cur_batch, $num_virtual_batches, $tempdir);
  print "Checking ".$wgs->[0][0]." and ".$wgs->[1][0]."\n";

  # Check if word graph files already exist
  if (!(-e $wgs->[0][0] && -e $wgs->[1][0])) {
    # Generate LNAs
    system("$AKUBINDIR/phone_probs -b $HMMMODEL -c $HMMMODEL.cfg -r $RECIPE -o $tempdir -B $num_virtual_batches -I $cur_batch $LNA_OPTIONS -i 1") && die "phone_probs failed\n";
    
    # Generate lattices
    system("$LATTICERECSCRIPT $HMMMODEL $LEXICON $RECIPE $tempdir $num_virtual_batches $cur_batch") && die "recognition failed\n";
    
    # Remove LNA files
    system("rm -f $tempdir/*.lna");

    $wg_generated = 1;
  }

  # Check if denominator hmmnets have been generated already
  if ($wg_generated || !(-e $wgs->[0][1] && -e $wgs->[1][1])) {
    # Generate hmmnets
    system("$SCRIPTDIR/make_den_fst.pl $USE_MORPHS $VOCABULARY $LMMODEL $RECIPE $tempdir $SCRIPTDIR $num_virtual_batches $cur_batch $LATTICE_THRESHOLD $LMSCALE") && die "hmmnet generation failed\n";
  }
}


sub load_recipe {
  my $recipe_file = shift(@_);
  my $batch_index = shift(@_);
  my $num_batches = shift(@_);
  my $temp_dir = shift(@_);
  my $target_lines;
  my $fh;
  my @recipe_lines;
  my $wgfile;
  my $denfile;
  my @result;
  
  open $fh, "< $recipe_file" || die "Could not open $recipe_file\n";
  @recipe_lines = <$fh>;
  close $fh;

  if ($num_batches <= 1) {
    $target_lines = $#recipe_lines;
  } else {
    $target_lines = int($#recipe_lines/$num_batches);
    $target_lines = 1 if ($target_lines < 1);
  }
  
  my $cur_index = 1;
  my $cur_line = 0;

  foreach my $line (@recipe_lines) {
    if ($num_batches > 1 && $cur_index < $num_batches) {
      if ($cur_line >= $target_lines) {
        $cur_index++;
        last if ($cur_index > $batch_index);
        $cur_line -= $target_lines;
      }
    }

    if ($num_batches <= 1 || $cur_index == $batch_index) {
      $line =~ /lna=(\S*)/;
      $wgfile = $temp_dir."/".$1.".wg";
      $line =~ /den\-hmmnet=(\S*)/;
      $denfile = $1;
      push(@result, [$wgfile, $denfile]) if ($cur_line == 0);
    }
    $cur_line++;
  }
  push(@result, [$wgfile, $denfile]);
  return \@result;
}
