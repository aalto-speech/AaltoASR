#!/usr/bin/perl

use locale;
use strict;
use ClusterManager;


## Model name ##
  my $BASE_ID = $ENV{'TRAIN_NAME'};
  defined($BASE_ID) || die("TRAIN_NAME environment variable needs to be set.");

## Path settings ##
  my $BINDIR = $ENV{'TRAIN_BINDIR'};
  defined($BINDIR) || die("TRAIN_BINDIR environment variable needs to be set.");
  my $SCRIPTDIR = $ENV{'TRAIN_SCRIPTDIR'};
  defined($SCRIPTDIR) || die("TRAIN_SCRIPTDIR environment variable needs to be set.");
  my $WORKDIR = $ENV{'TRAIN_DIR'};
  defined($WORKDIR) || die("TRAIN_DIR environment variable needs to be set.");

## Training file list ##
  my $RECIPE = $ENV{'TRAIN_RECIPE'};
  defined($RECIPE) || die("TRAIN_RECIPE environment variable needs to be set.");

## Initial model names ##
  my $init_model = $ENV{'TRAIN_IM'};
  defined($init_model) || die("TRAIN_IM environment variable needs to be set.");

## Batch settings ##
  # Number of batches, maximum number of parallel jobs
  my $NUM_BATCHES = $ENV{'TRAIN_BATCHES'};
  $NUM_BATCHES = 20 if !defined($NUM_BATCHES);
  my $BATCH_PRIORITY = 0; # Not used currently.
  my $BATCH_MAX_MEM = 2000; # In megabytes
  # Note that you may need memory if the training data contains
  # e.g. long utterances! If too little memory is reserved, unexpected
  # termination of the training may occur.

## Train/Baum-Welch settings ##
  my $USE_HMMNETS = 1; # If 0, the script must call align appropriately
  my $FORWARD_BEAM = 15;
  my $BACKWARD_BEAM = 200;
  my $AC_SCALE = 1; # Acoustic scaling (For ML 1, for MMI 1/(LMSCALE/lne(10)))
  my $ML_STATS_MODE = "--ml";
  my $ML_ESTIMATE_MODE = "--ml";

## HMMNET options ##
  my $MORPH_HMMNETS = 0; # True (1) if HMMNETs are not based on words
  my $LEX_FILE=$ENV{'TRAIN_LEX'};
  defined($LEX_FILE) || die("TRAIN_LEX environment variable needs to be set.");
  my $TRN_FILE=$ENV{'TRAIN_TRN'};
  defined($TRN_FILE) || die("TRAIN_TRN environment variable needs to be set.");

## Alignment settings ##
  my $ALIGN_WINDOW = 4000;
  my $ALIGN_BEAM = 600;
  my $ALIGN_SBEAM = 100;

## Misc settings ##
  my $VERBOSITY = 1;

  # NOTE: If you plan to recompile executables at the same time as running
  # them, it is a good idea to copy the old binaries to a different directory.
  my $COPY_BINARY_TO_WORK = 1;

## Ignore some nodes if SLURM_EXCLUDE_NODES environment variable is set ##
  my $EXCLUDE_NODES = $ENV{'SLURM_EXCLUDE_NODES'};
  $EXCLUDE_NODES = '' if !defined($EXCLUDE_NODES);


######################################################################
# Alignment script begins
######################################################################

# Create own working directory
my $tempdir = "$WORKDIR/temp";
mkdir $WORKDIR;
mkdir $tempdir;
chdir $tempdir || die("Could not chdir to $tempdir");

if ($COPY_BINARY_TO_WORK > 0) {
    copy_binary_to_work($BINDIR, "$tempdir/bin");
}

generate_hmmnet_files($init_model, $tempdir);

if ($USE_HMMNETS) {
  align_hmmnets($tempdir, $init_model, $RECIPE);
} else {
  align($tempdir, $init_model, $RECIPE);
}


sub copy_binary_to_work {
    my $orig_bin_dir = shift(@_);
    my $new_bin_dir = shift(@_);

    mkdir $new_bin_dir;
    system("cp ${orig_bin_dir}/estimate ${new_bin_dir}");
    system("cp ${orig_bin_dir}/align ${new_bin_dir}");
    system("cp ${orig_bin_dir}/tie ${new_bin_dir}");
    system("cp ${orig_bin_dir}/stats ${new_bin_dir}");
    system("cp ${orig_bin_dir}/vtln ${new_bin_dir}");
    system("cp ${orig_bin_dir}/dur_est ${new_bin_dir}");
    system("cp ${orig_bin_dir}/gconvert ${new_bin_dir}");
    system("cp ${orig_bin_dir}/gcluster ${new_bin_dir}");
    system("cp ${orig_bin_dir}/phone_probs ${new_bin_dir}");
    system("cp ${orig_bin_dir}/mllr ${new_bin_dir}");
    system("cp ${orig_bin_dir}/combine_stats ${new_bin_dir}");
    $BINDIR = $new_bin_dir;
}


sub generate_hmmnet_files {
  my $im = shift(@_);
  my $temp_dir = shift(@_);

  my $new_temp_dir = "$temp_dir/hmmnets";
  mkdir $new_temp_dir;
  chdir $new_temp_dir || die("Could not chdir to $new_temp_dir");

  my $cm = ClusterManager->new;
  $cm->{"identifier"} = "hmmnets_${BASE_ID}";
  $cm->{"run_dir"} = $new_temp_dir;
  $cm->{"log_dir"} = $new_temp_dir;
  $cm->{"run_time"} = 2000;
  $cm->{"mem_req"} = 1000;
  if ($NUM_BATCHES > 0) {
    $cm->{"first_batch"} = 1;
    $cm->{"last_batch"} = $NUM_BATCHES;
    $cm->{"run_time"} = 239;
  }
  $cm->{"failed_batch_retry_count"} = 1;
  $cm->{"exclude_nodes"} = $EXCLUDE_NODES;

  my $batch_options = "";
  $batch_options = "-B $NUM_BATCHES -I \$BATCH" if ($NUM_BATCHES > 0);

  # Create hmmnets from TRN transcripts with a lexicon and
  # alternative paths e.g. for silences:
  my $morph_switch = "";
  if ($MORPH_HMMNETS > 0) {
    $morph_switch = "-m ${LEX_FILE}.voc";
  }
  my $trn_switch = "-t $TRN_FILE";
  # Construct helper FSTs (L.fst, C.fst, H.fst, optional_silence.fst and
  # end_mark.fst) and vocabulary file.
  # Assumes that the current directory is $temp_dir!
  if ($MORPH_HMMNETS > 0) {
    $morph_switch = "-m";
  }
  system("$SCRIPTDIR/build_helper_fsts.sh $morph_switch -s $SCRIPTDIR $LEX_FILE $im.ph");

  # Use real FST processing in create_hmmnets.pl to create hmmnets
  # with alternative paths for pronunciations and silences
  $cm->submit("$SCRIPTDIR/create_hmmnets.pl -n -r $RECIPE $morph_switch $trn_switch -T $new_temp_dir -F $new_temp_dir -D $BINDIR -s $SCRIPTDIR $batch_options\n", "");

  chdir($temp_dir);
}


sub align {
  my $temp_dir = shift(@_);
  my $model = shift(@_);
  my $recipe = shift(@_);
  my $spkc_file = shift(@_);
  my $batch_options = "";
  my $spkc_switch = "";
  $spkc_switch = "-S $spkc_file" if ($spkc_file ne "");

  my $new_temp_dir = "$temp_dir/align";
  mkdir $new_temp_dir;
  chdir $new_temp_dir || die("Could not chdir to $new_temp_dir");

  my $cm = ClusterManager->new;
  $cm->{"identifier"} = "align_${BASE_ID}";
  $cm->{"run_dir"} = $new_temp_dir;
  $cm->{"log_dir"} = $new_temp_dir;
  $cm->{"mem_req"} = $BATCH_MAX_MEM;
  if ($NUM_BATCHES > 0) {
    $cm->{"first_batch"} = 1;
    $cm->{"last_batch"} = $NUM_BATCHES;
  }
  $cm->{"priority"} = $BATCH_PRIORITY;
  $cm->{"failed_batch_retry_count"} = 1;
  $cm->{"exclude_nodes"} = $EXCLUDE_NODES;

  $batch_options = "-B $NUM_BATCHES -I \$BATCH" if ($NUM_BATCHES > 0);
  $cm->submit("$BINDIR/align -b $model -c $model.cfg -r $recipe --swins $ALIGN_WINDOW --beam $ALIGN_BEAM --sbeam $ALIGN_SBEAM $spkc_switch $batch_options -i $VERBOSITY\n", "");

  chdir($temp_dir);
}


sub align_hmmnets {
  my $temp_dir = shift(@_);
  my $model = shift(@_);
  my $recipe = shift(@_);
  my $spkc_file = shift(@_);
  my $batch_options = "";
  my $spkc_switch = "";
  $spkc_switch = "-S $spkc_file" if ($spkc_file ne "");

  my $new_temp_dir = "$temp_dir/align";
  mkdir $new_temp_dir;
  chdir $new_temp_dir || die("Could not chdir to $new_temp_dir");

  my $cm = ClusterManager->new;
  $cm->{"identifier"} = "align_${BASE_ID}";
  $cm->{"run_dir"} = $new_temp_dir;
  $cm->{"log_dir"} = $new_temp_dir;
  $cm->{"run_time"} = 2000;
  $cm->{"mem_req"} = $BATCH_MAX_MEM;
  if ($NUM_BATCHES > 0) {
    $cm->{"first_batch"} = 1;
    $cm->{"last_batch"} = $NUM_BATCHES;
    $cm->{"run_time"} = 239;
  }
  $cm->{"priority"} = $BATCH_PRIORITY;
  $cm->{"failed_batch_retry_count"} = 1;
  $cm->{"exclude_nodes"} = $EXCLUDE_NODES;

  $batch_options = "-B $NUM_BATCHES -I \$BATCH" if ($NUM_BATCHES > 0);
  $cm->submit("$BINDIR/stats -b $model -c $model.cfg -r $recipe -H --ml -M vit -a -n -o /dev/null $spkc_switch $batch_options -i $VERBOSITY\n", "");
}
