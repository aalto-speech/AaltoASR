#!/usr/bin/perl
#
# Generates LNA files with possible Gaussian clustering, recognizes
# the utterances and deletes the LNAs in the end.  Enables parallel
# operation for both LNA generation and recognition.
#

use lib '/share/puhe/x86_64/lib'; # For GridManager (or use aku/scripts dir)
use locale;
use strict;
use GridManager;


# Arguments:
my $TASK_ID = shift; # Unique identifier, also the name of the lna directory
my $outdir = shift;
my $tempdir = shift;
my $lna_recipe=shift;
my $model = shift;
my $cfg = shift;


# Fix these paths if necessary!
my $lna_outdir = "/share/work/".$ENV{"USER"}."/lnas/".$TASK_ID;
my $BINDIR="/home/".$ENV{"USER"}."/aku";
my $DECODERDIR="/home/".$ENV{"USER"}."/decoder";

# Batch settings
my $NUM_LNA_BATCHES = 2; # Number of processes in parallel
my $NUM_REC_BATCHES = 1;
my $BATCH_PRIORITY = 0;

# Gaussian clustering options
my $NUM_GAUSS_CLUSTERS = 1000; # 0 if no clustering
my $GAUSS_EVAL_RATIO = 0.15;


chdir $tempdir || die("Could not chdir to $tempdir");

if ($NUM_GAUSS_CLUSTERS > 0 && !(-e $model.".gcl")) {
  cluster_gaussians($tempdir, $model);
}

generate_lnas($tempdir, $model, $cfg, $lna_recipe, $lna_outdir);

recognize($outdir, $tempdir, $model, $lna_recipe, $na_outdir);

system("rm -rf $lna_outdir");


sub cluster_gaussians {
  my $temp_dir = shift(@_);
  my $im = shift(@_);

  my $gm = GridManager->new;
  $gm->{"identifier"} = "gcluster_${TASK_ID}";
  $gm->{"run_dir"} = $temp_dir;
  $gm->{"log_dir"} = $temp_dir;
  $gm->submit("$BINDIR/gcluster -g $im.gk -o $im.gcl -C $NUM_GAUSS_CLUSTERS");
  if (!(-e $im.".gcl")) {
    die("Error in Gaussian clustering\n");
  }
}

sub generate_lnas {
  my $temp_dir = shift(@_);
  my $model = shift(@_);
  my $cfg = shift(@_);
  my $recipe = shift(@_);
  my $out_dir = shift(@_);
  my $spkc_file = shift(@_);
  my $batch_options = "";
  my $cluster_options = "";

  my $spkc_switch = "";
  $spkc_switch = "-S $spkc_file" if ($spkc_file ne "");

  mkdir $out_dir;

  my $gm = GridManager->new;
  $gm->{"identifier"} = "lna_${TASK_ID}";
  $gm->{"run_dir"} = $temp_dir;
  $gm->{"log_dir"} = $temp_dir;
  if ($NUM_LNA_BATCHES > 0) {
    $gm->{"first_batch"} = 1;
    $gm->{"last_batch"} = $NUM_LNA_BATCHES;
    $batch_options = "-B $NUM_LNA_BATCHES -I \$BATCH"
  }
  $gm->{"priority"} = $BATCH_PRIORITY;
  #$gm->{"emergency_shutdown"} = 0;
  $gm->{"failed_batch_retry_count"} = 1;
  $cluster_options = "-C $model.gcl --eval-ming $GAUSS_EVAL_RATIO" if ($NUM_GAUSS_CLUSTERS);
  #self->{"qsub_options"} = self->{"qsub_options"}." -soft -q helli.q";
  $gm->submit("$BINDIR/phone_probs -b $model -c $cfg -r $recipe -o $out_dir $spkc_switch $batch_options -i 1 $cluster_options");
}

sub recognize {
  my $out_dir = shift(@_);
  my $temp_dir = shift(@_);
  my $model = shift(@_);
  my $recipe = shift(@_);
  my $lna_outdir = shift(@_);
  my $batch_options = "";
  my $name_tail = ".log";

  my $gm = GridManager->new;
  $gm->{"identifier"} = "rec_${TASK_ID}";
  $gm->{"run_dir"} = $temp_dir;
  $gm->{"log_dir"} = $temp_dir;
  $batch_options = "1 1";
  if ($NUM_REC_BATCHES > 1) {
    $gm->{"first_batch"} = 1;
    $gm->{"last_batch"} = $NUM_REC_BATCHES;
    $batch_options = "$NUM_REC_BATCHES \$BATCH";
    $name_tail = "_part_\$BATCH".$name_tail;
  }
  $gm->{"priority"} = $BATCH_PRIORITY;
  $gm->{"failed_batch_retry_count"} = 0;
  #self->{"qsub_options"} = self->{"qsub_options"}." -q helli.q";
  $gm->submit("$DECODERDIR/rec_recipe2.py $model $recipe $lna_outdir $batch_options > $out_dir/${TASK_ID}${name_tail}");
}
