#!/usr/bin/perl

# This script creates the denominator hmmnets required for discriminative
# training. Note that this will also generate the normal numerator hmmnets
# to ensure conformity of the two hmmnets. If you want to disable the
# writing of numerator hmmnets, remove the -n switch from create_hmmnets.pl
# call.
# Naturally the recipe file needs to have both hmmnet and den-hmmnet fields
# set and those paths must exist (see scripts/make_recipe_paths.pl for
# creating the paths).


# This is a TEMPLATE SCRIPT, you need to modify the paths, scripts, and
# input files according to your needs!


use lib '/share/puhe/scripts/cluster'; # For CondorManager
use locale;
use strict;
use CondorManager;


# Model name
my $ID="speecon_all_den";

# Path settings
my $AKUBINDIR="/home/".$ENV{"USER"}."/aku";
my $SCRIPTDIR="/home/".$ENV{"USER"}."/aku/scripts";
my $HMMDIR="/share/puhe/".$ENV{"USER"}."/hmms";
my $workdir="/share/work/".$ENV{"USER"}."/aku_work";

# Training file list, writes lattices to where hmmnet and den-hmmnet keys point
my $RECIPE="/full/path/of/your.recipe";

# Script for generating word graphs (recognition of the recipe).
# Has to have language models hard-wired in the script!
# The script takes the following arguments (listed in order):
#   1. HMM model
#   2. Recipe
#   3. LNA-path
# The output lattice is written to LNAPath/LNAFile.wg
my $LATTICERECSCRIPT="/share/puhe/jpylkkon/aku_test/speecon_rec_lattice.py";

my $HMMMODEL="$HMMDIR/speecon_ml_20";
my $LEXICON="/share/work/jpylkkon/lm/morph19k_fillers.lex";

# Note that for discriminative training, it is usually beneficial to use
# a minimal language model to rescore the lattices, e.g. a unigram model.
my $LMMODEL="/share/work/jpylkkon/lm/morph19k_1gram.bin";

# Enable morph processing by defining a morph vocabulary, or leave empty
# for word based processing.
my $MORPH_VOC = "/share/work/jpylkkon/lm/morph19k_fillers.voc";

# TRN file for transcriptions. If not defined, reads the PHN files,
# but that is only feasible for grapheme-based models (e.g. Finnish).
my $TRANSCRIPT_FILE = "";

# Batch settings
my $NUM_BATCHES = 50; # Number of processes in parallel
my $NUM_BLOCKS = 1; # In how many blocks the data is generated in one batch

my $LATTICE_THRESHOLD = 0.00000001; # Lattice pruning threshold, [0, 1)
my $LMSCALE = 30;

my $LNA_OPTIONS = "\'--clusters ${HMMMODEL}.gcl --eval-ming=0.15\'";


## Execution part begins here  ##

my $tempdir = $workdir."/".$ID;
mkdir $tempdir;
chdir $tempdir || die("Could not chdir to $tempdir");

my $morph_switch = "";
$morph_switch = "-m" if (length($MORPH_VOC) > 0);
system("$SCRIPTDIR/build_helper_fsts.sh $morph_switch -s $SCRIPTDIR $LEXICON $HMMMODEL.ph");

my $cm = CondorManager->new;
$cm->{"identifier"} = $ID;
$cm->{"run_dir"} = $tempdir;
$cm->{"log_dir"} = $tempdir;
$cm->{"first_batch"} = 1;
$cm->{"last_batch"} = $NUM_BATCHES;
$cm->{"failed_batch_retry_count"} = 0;

my $tr_handling = "";
if (length($MORPH_VOC) > 0) {
  $tr_handling = "-m $MORPH_VOC";
}
if (length($TRANSCRIPT_FILE) > 0) {
    $tr_handling = $tr_handling." -t $TRANSCRIPT_FILE";
}

$cm->submit("$SCRIPTDIR/create_hmmnets.pl -n -d -r $RECIPE -B $NUM_BATCHES -I \$BATCH -F $tempdir -T $tempdir -p $LATTICE_THRESHOLD -l $LMMODEL -L $LMSCALE -b $HMMMODEL -c $HMMMODEL.cfg -D $AKUBINDIR -s $SCRIPTDIR -P $LNA_OPTIONS -R $LATTICERECSCRIPT $tr_handling", "");
