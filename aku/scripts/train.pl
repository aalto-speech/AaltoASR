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
  my $HMMDIR = "$WORKDIR/hmm";

## Training file list ##
  my $RECIPE = $ENV{'TRAIN_RECIPE'};
  defined($RECIPE) || die("TRAIN_RECIPE environment variable needs to be set.");

## Initial model names ##
  # Now initial model is created in tying, and we use only the .cfg file
  # of the original model.
  my $init_model = "$HMMDIR/$BASE_ID";
  my $init_cfg = "$HMMDIR/init.cfg";

## Batch settings ##
  # Number of batches, maximum number of parallel jobs
  my $NUM_BATCHES = $ENV{'TRAIN_BATCHES'};
  $NUM_BATCHES = 20 if !defined($NUM_BATCHES);
  my $BATCH_PRIORITY = 0; # Not used currently.
  my $BATCH_MAX_MEM = 2000; # In megabytes
  # Note that you may need memory if the training data contains
  # e.g. long utterances! If too little memory is reserved, unexpected
  # termination of the training may occur. On Condor there are currently
  # no memory restrictions.

## Train/Baum-Welch settings ##
  my $USE_HMMNETS = 1; # If 0, the script must call align appropriately
  my $FORWARD_BEAM = 15;
  my $BACKWARD_BEAM = 200;
  my $AC_SCALE = 1; # Acoustic scaling (For ML 1, for MMI 1/(LMSCALE/lne(10)))
  my $ML_STATS_MODE = "--ml";
  my $ML_ESTIMATE_MODE = "--ml";

## Alignment settings ##
  my $ALIGN_WINDOW = 4000;
  my $ALIGN_BEAM = 1000;
  my $ALIGN_SBEAM = 100;

## Context phone tying options ##
  my $TIE_USE_OUT_PHN = 0; # 0=read transcript field, 1=read alignment field
  my $TIE_RULES = $ENV{'TRAIN_TIERULES'};
  defined($TIE_RULES) || die("TRAIN_TIERULES environment variable needs to be set.");
  my $TIE_MIN_COUNT = 1500; # Minimum number of features per state
  my $TIE_MIN_GAIN = 5000;  # Minimum loglikelihood gain for a state split
  my $TIE_MAX_LOSS = 5000;  # Maximum loglikelihood loss for merging states

## Gaussian splitting options ##
  my $SPLIT_MIN_OCCUPANCY = 200; # Accumulated state probability
  my $SPLIT_MAX_GAUSSIANS = 80; # Per state
  # If $SPLIT_TARGET_GAUSSIANS > 0, it defines the Gaussian splitting instead
  # of $SPLIT_MIN_OCCUPANCY.
  my $SPLIT_TARGET_GAUSSIANS = $ENV{'TRAIN_GAUSSIANS'};
  $SPLIT_TARGET_GAUSSIANS = -1 if !defined($SPLIT_TARGET_GAUSSIANS);
  my $SPLIT_ALPHA = 0.3;  # Smoothing power for occupancies
  my $GAUSS_REMOVE_THRESHOLD = 0.001; # Mixture component weight threshold

## Minimum variance ##
  my $MINVAR = 0.1;

## Gaussian clustering options ##
  my $NUM_GAUSS_CLUSTERS = $ENV{'TRAIN_CLUSTERS'};
  $NUM_GAUSS_CLUSTERS = 1000 if !defined($NUM_GAUSS_CLUSTERS); # 0 if no clustering
  my $GAUSS_EVAL_RATIO = 0.1;

## MLLT options ##
  my $mllt_start_iter = 15; # At which iteration MLLT estimation should begin
  my $mllt_frequency = 2; # How many EM iterations between MLLT estimation
  my $MLLT_MODULE_NAME = "transform";

## Training iterations ##
  my $num_ml_train_iter = 22;
  my $split_frequency = 2; # How many EM iterations between Gaussian splits
  my $split_stop_iter = 18; # Iteration after which no more splits are done

## Adaptation settings ##
  my $VTLN_MODULE = "vtln";
  my $MLLR_MODULE = "mllr";
  my $SPKC_FILE = ""; # For initialization see e.g. $SCRIPTDIR/vtln_default.spkc

## Misc settings ##
  # Remove duration models for states that start with an underscore
  my $REMOVE_DUR_MODELS = 1;
  my $VERBOSITY = 1;
  # NOTE: If you plan to recompile executables at the same time as running
  # them, it is a good idea to copy the old binaries to a different directory.
  my $COPY_BINARY_TO_WORK = 1;
  my $SAVE_STATISTICS = 0; # Save the statistics files in iteration directories

## Ignore some nodes if SLURM_EXCLUDE_NODES environment variable is set ##
  my $EXCLUDE_NODES = $ENV{'SLURM_EXCLUDE_NODES'};
  $EXCLUDE_NODES = '' if !defined($EXCLUDE_NODES);


######################################################################
# Training script begins
######################################################################

# Create own working directory
my $tempdir = "$WORKDIR/temp";
mkdir $WORKDIR;
mkdir $tempdir;
mkdir $HMMDIR;
chdir $tempdir || die("Could not chdir to $tempdir");

if ($COPY_BINARY_TO_WORK > 0) {
    copy_binary_to_work($BINDIR, "$tempdir/bin");
}

# Generate initial model by context phone tying using existing alignments
context_phone_tying($init_model, $init_cfg);

# Convert the generated full covariance model to diagonal model
convert_full_to_diagonal($init_model);

# Create the hmmnet files
generate_hmmnet_files($init_model, $tempdir);

# ML/EM training
my $ml_model;
$ml_model=ml_train($tempdir, 1, $num_ml_train_iter, $init_model, $init_cfg,
                   $mllt_start_iter, $mllt_frequency,
                   $split_frequency, $split_stop_iter);
my $om = $ml_model;

# Estimate duration model
if ($USE_HMMNETS) {
  align_hmmnets($tempdir, $om, $RECIPE);
} else {
  align($tempdir, $om, $RECIPE);
}
estimate_dur_model($om);

# VTLN
# align($tempdir, $om, $RECIPE);
# estimate_vtln($tempdir, $om, $RECIPE, $om.".spkc");

# # MLLR
# estimate_mllr($tempdir, $om, $RECIPE, $om.".spkc");

# Cluster the Gaussians
 if ($NUM_GAUSS_CLUSTERS > 0) {
   cluster_gaussians($om);
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


sub context_phone_tying {
  my $out_model = shift(@_);
  my $im_cfg = shift(@_);
  my $phn_flag = "";
  $phn_flag = "-O" if ($TIE_USE_OUT_PHN);

  my $cm = ClusterManager->new;
  $cm->{"identifier"} = "tie_".$BASE_ID;
  $cm->{"run_dir"} = $tempdir;
  $cm->{"log_dir"} = $tempdir;
  $cm->{"run_time"} = 239;
  $cm->{"mem_req"} = 4000;
  $cm->{"exclude_nodes"} = $EXCLUDE_NODES;

  $cm->submit("$BINDIR/tie -c $im_cfg -o $out_model -r $RECIPE $phn_flag -u $TIE_RULES --count $TIE_MIN_COUNT --sgain $TIE_MIN_GAIN --mloss $TIE_MAX_LOSS -i $VERBOSITY\n", "");

  if (!(-e $out_model.".ph")) {
    die "Error in context phone tying\n";
  }
}


sub convert_full_to_diagonal {
  my $im = shift(@_);
  my $gk = $im.".gk";
  my $gk_backup = $im."_full.gk";
  
  system("mv $gk $gk_backup");
  system("$BINDIR/gconvert -g $gk_backup -o $gk -d > $tempdir/gconvert.stdout 2> $tempdir/gconvert.stderr\n");
}


sub ml_train {
  my $temp_dir = shift(@_);
  my $iter_init = shift(@_);
  my $iter_end = shift(@_);

  my $im = shift(@_);
  my $im_cfg = shift(@_);

  my $mllt_start = shift(@_);
  my $mllt_frequency = shift(@_);
  my $mllt_flag = 0;

  my $split_frequency = shift(@_);
  my $split_stop_iter = shift(@_);
  my $split_flag;

  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime(time);
  my $dstring = "$mday.".($mon+1).".".(1900+$year);
  my $model_base = "$HMMDIR/${BASE_ID}_${dstring}";

  my $stats_list_file = "statsfiles.lst";
  my $batch_info;

  for (my $i = $iter_init; $i <= $iter_end ; $i ++) {

    print "ML iteration ".$i."\n" if ($VERBOSITY > 0);
    my $om = $model_base."_".$i;

    my $log_dir = "$temp_dir/ml_iter_".$i;
    mkdir $log_dir;

    $mllt_flag = 0;
    $mllt_flag = 1 if ($mllt_start && $i >= $mllt_start &&
		       (($i-$mllt_start) % $mllt_frequency) == 0);
    
    collect_stats("ml_$i", $temp_dir, $log_dir, $im, $im_cfg, $stats_list_file,
                  $ML_STATS_MODE, $mllt_flag);

    my $fh;
    open($fh, $stats_list_file) || die("Could not open file $stats_list_file!");
    my @stats_files=<$fh>;
    close($fh);

    if ($SAVE_STATISTICS) {
      my $cur_file;
      foreach $cur_file (@stats_files) {
        chomp $cur_file;
        system("cp $cur_file.* $log_dir");
      }
      system("cp $stats_list_file $log_dir");
    }

    $split_flag = 0;
    $split_flag = 1 if ($split_frequency && $i < $split_stop_iter &&
                        (($i-1) % $split_frequency) == 0);
    estimate_model($log_dir, $im, $im_cfg, $om, $stats_list_file, $MINVAR,
                   $ML_ESTIMATE_MODE, $mllt_flag, $split_flag);
    
    # Check the models were really created
    if (!(-e $om.".ph")) {
      die "Error in training, no models were written\n";
    }

    # Remove the statistics files
    my $cur_file;
    foreach $cur_file (@stats_files) {
      chomp $cur_file;
      system("rm $cur_file.*");
    }
    system("rm $stats_list_file");

    # Read input from previously written model
    $im = $om;
    $im_cfg = $om.".cfg";
  }
  return $im;
}


sub collect_stats {
  my $id = shift(@_);
  my $temp_dir = shift(@_);
  my $log_dir = shift(@_);
  my $model_base = shift(@_);
  my $cfg = shift(@_);
  my $stats_list_file = shift(@_);
  my $stats_mode = shift(@_);
  my $mllt_flag = shift(@_);
  my $batch_options = "";
  my $spkc_switch = "";
  my $bw_option = "";
  my $mllt_option = "";
  $bw_option = "-H" if ($USE_HMMNETS);
  $spkc_switch = "-S $SPKC_FILE" if ($SPKC_FILE ne "");
  $mllt_option = "--mllt" if ($mllt_flag);

  my $cm = ClusterManager->new;
  $cm->{"identifier"} = $id;
  $cm->{"run_dir"} = $temp_dir;
  $cm->{"log_dir"} = $log_dir;
  $cm->{"run_time"} = 2000;
  if ($NUM_BATCHES > 0) {
    $cm->{"first_batch"} = 1;
    $cm->{"last_batch"} = $NUM_BATCHES;
    $cm->{"run_time"} = 239;
  }
  $cm->{"priority"} = $BATCH_PRIORITY;
  $cm->{"mem_req"} = $BATCH_MAX_MEM;
  $cm->{"failed_batch_retry_count"} = 1;
  $cm->{"exclude_nodes"} = $EXCLUDE_NODES;

  $batch_options = "-B $NUM_BATCHES -I \$BATCH" if ($NUM_BATCHES > 0);

  my $failed_batches_file = "${id}_failed_batches.lst";
  unlink($failed_batches_file);

  $cm->submit("$BINDIR/stats -b $model_base -c $cfg -r $RECIPE $bw_option -o temp_stats_\$BATCH $stats_mode -F $FORWARD_BEAM -W $BACKWARD_BEAM -A $AC_SCALE -t $spkc_switch $batch_options -i $VERBOSITY $mllt_option\n".
"if [ \"\$?\" -ne \"0\" ]; then echo \$BATCH >> $failed_batches_file; exit 1; fi\n",
# Epilog script:
"if [ -f ${id}_stats.gks ] ; then\n".
"  if [ ! -f ${id}_stats_list ] ; then\n".
"    echo ${id}_stats > ${id}_stats_list\n".
"  fi\n".
"  echo temp_stats_\$BATCH >> ${id}_stats_list\n".
"  l=`wc -l ${id}_stats_list | cut -f 1 -d ' '`\n".
"  s=`ls -1 temp_stats_*.lls | wc -l | cut -f 1 -d ' '`\n".
"  if [[ \$l -gt \$s || \$RUNNING_JOBS -eq 0 ]]; then\n".
"    $BINDIR/combine_stats -b $model_base -L ${id}_stats_list -o ${id}_stats\n".
"    if [ \"\$?\" -ne \"0\" ]; then rm ${id}_stats_list; rm temp_stats_\$BATCH.*; exit 1; fi\n".
"    for i in `tail -n +2 ${id}_stats_list`; do rm \$i.*; done\n".
"    rm ${id}_stats_list\n".
"  fi\n".
"else\n".
"  mv temp_stats_\$BATCH.gks ${id}_stats.gks\n".
"  mv temp_stats_\$BATCH.mcs ${id}_stats.mcs\n".
"  mv temp_stats_\$BATCH.phs ${id}_stats.phs\n".
"  mv temp_stats_\$BATCH.lls ${id}_stats.lls\n".
"fi\n");

  die "Some batches failed (see $temp_dir/$failed_batches_file)\n" if (-e $failed_batches_file);

  # Write the list of statistics files
  system("echo ${id}_stats > $stats_list_file");
}


sub estimate_model {
  my $log_dir = shift(@_);
  my $im = shift(@_);
  my $im_cfg = shift(@_);
  my $om = shift(@_);
  my $stats_list_file = shift(@_);
  my $minvar = shift(@_);
  my $estimate_mode = shift(@_);
  my $mllt_flag = shift(@_);
  my $split_flag = shift(@_);
  my $extra_options = "";

  $extra_options = "--mllt $MLLT_MODULE_NAME" if ($mllt_flag);
  $extra_options = $extra_options." --mremove $GAUSS_REMOVE_THRESHOLD" if ($GAUSS_REMOVE_THRESHOLD > 0);
  if ($split_flag) {
    if ($SPLIT_TARGET_GAUSSIANS > 0) {
      $extra_options = $extra_options." --split --numgauss $SPLIT_TARGET_GAUSSIANS --maxmixgauss $SPLIT_MAX_GAUSSIANS";
      $extra_options = $extra_options." --minocc $SPLIT_MIN_OCCUPANCY" if ($SPLIT_MIN_OCCUPANCY > 0);
    } else {
      $extra_options = $extra_options." --split --minocc $SPLIT_MIN_OCCUPANCY --maxmixgauss $SPLIT_MAX_GAUSSIANS";
    }
    if (defined $SPLIT_ALPHA && $SPLIT_ALPHA > 0) {
      $extra_options = $extra_options." --splitalpha $SPLIT_ALPHA";
    }
  }

  system("$BINDIR/estimate -b $im -c $im_cfg -L $stats_list_file -o $om -t -i $VERBOSITY --minvar $minvar $estimate_mode -s ${BASE_ID}_summary $extra_options > $log_dir/estimate.stdout 2> $log_dir/estimate.stderr");
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

  # Create hmmnets directly from PHN files, without lexicon
  # or alternative paths e.g. for silences:
  $cm->submit("$SCRIPTDIR/create_hmmnets.pl -n -o -r $RECIPE -b $im -T $new_temp_dir -D $BINDIR -s $SCRIPTDIR $batch_options");

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



# This version runs Baum-Welch
sub estimate_mllr {
  my $temp_dir = shift(@_);
  my $model = shift(@_);
  my $recipe = shift(@_);
  my $out_file = shift(@_);
  my $temp_out;
  my $batch_options = "";

  my $cm = ClusterManager->new;
  $cm->{"identifier"} = "mllr_${BASE_ID}";
  $cm->{"run_dir"} = $temp_dir;
  $cm->{"log_dir"} = $temp_dir;
  $cm->{"run_time"} = 2000;
  $cm->{"mem_req"} = $BATCH_MAX_MEM;
  if ($NUM_BATCHES > 0) {
    $cm->{"first_batch"} = 1;
    $cm->{"last_batch"} = $NUM_BATCHES;
    $cm->{"run_time"} = 239;
  }
  $cm->{"priority"} = $BATCH_PRIORITY;
  $cm->{"mem_req"} = $BATCH_MAX_MEM; # No large mem requirements
  $cm->{"failed_batch_retry_count"} = 1;
  $cm->{"exclude_nodes"} = $EXCLUDE_NODES;

  $temp_out = $out_file;
  if ($NUM_BATCHES > 1) {
    $batch_options = "-B $NUM_BATCHES -I \$BATCH";
    $temp_out = "mllr_temp_\$BATCH.spkc";
  }
  $cm->submit("$BINDIR/mllr -b $model -c $model.cfg -r $recipe -H -F $FORWARD_BEAM -W $BACKWARD_BEAM -M $MLLR_MODULE -S $SPKC_FILE -o $temp_out $batch_options -i $VERBOSITY\n", "");

  if ($NUM_BATCHES > 1) {
    system("cat mllr_temp_*.spkc > $out_file") && die("mllr estimation failed");
  }
}


# NOTE: Uses alignments
sub estimate_vtln {
  my $temp_dir = shift(@_);
  my $model = shift(@_);
  my $recipe = shift(@_);
  my $out_file = shift(@_);
  my $temp_out;
  my $batch_options = "";

  my $cm = ClusterManager->new;
  $cm->{"identifier"} = "vtln_${BASE_ID}.sh";
  $cm->{"run_dir"} = $temp_dir;
  $cm->{"log_dir"} = $temp_dir;
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

  $temp_out = $out_file;
  if ($NUM_BATCHES > 1) {
    $batch_options = "-B $NUM_BATCHES -I \$BATCH";
    $temp_out = "vtln_temp_\$BATCH.spkc";
  }
  $cm->submit("$BINDIR/vtln -b $model -c $model.cfg -r $recipe -O -v $VTLN_MODULE -S $SPKC_FILE -o $temp_out $batch_options -i $VERBOSITY\n", "");
  if ($NUM_BATCHES > 1) {
    system("cat vtln_temp_*.spkc > $out_file") && die("vtln estimation failed");
  }
}



# NOTE: Uses alignments
sub estimate_dur_model {
  my $om = shift(@_);
  system("$BINDIR/dur_est -p $om.ph -r $RECIPE -O --gamma $om.dur > $tempdir/dur_est.stdout 2> $tempdir/dur_est.stderr");
  if ($REMOVE_DUR_MODELS) {
    my @states = `grep --no-group-separator -A1 -e "^[^\+]*_[^\-]*\$" $om.ph | grep -v "_" | cut -f 3- -d" " | sed "s/ /\\n/g"`;
    s{^\s+|\s+$}{}g foreach @states;
    foreach (@states) {
      `sed -i "s/^$_ .*\$/$_ 0.000 0.000/" $om.dur`
    }
  }
  if (!(-e $om.".dur")) {
    die("Error estimating duration models\n");
  }
}


sub cluster_gaussians {
  my $im = shift(@_);
  system("$BINDIR/gcluster -g $im.gk -o $im.gcl -C $NUM_GAUSS_CLUSTERS -i $VERBOSITY > $tempdir/gcluster.stdout 2> $tempdir/gcluster.stderr");
  if (!(-e $im.".gcl")) {
    die("Error in Gaussian clustering\n");
  }
}
