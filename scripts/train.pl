#!/usr/bin/perl

# DO NOT USE THIS SCRIPT DIRECTLY! Copy it to your own directory and
# modify appropriately. You MUST modify at least the path settings,
# training file list and BASE_ID/initial model names!

# Run this script at stimulus.hut.fi, it uses GridEngine for scheduling
# the jobs to cluster machines.

use lib '/share/puhe/x86_64/lib'; # For GridManager (or use aku/scripts dir)
use locale;
use strict;
use GridManager;


# Model name
my $BASE_ID="mfcc";

# Path settings
my $BINDIR="/home/".$ENV{"USER"}."/aku";
my $SCRIPTDIR="$BINDIR/scripts";
my $HMMDIR="/share/puhe/".$ENV{"USER"}."/hmms";
my $workdir="/share/work/".$ENV{"USER"}."/aku_work";
my $lna_outdir = "/share/work/".$ENV{"USER"}."/lnas";

# Training file list
my $RECIPE="/share/puhe/jpylkkon/speecon_new/speecon_new_train.recipe";
my $lna_recipe="/share/puhe/jpylkkon/speecon_new/speecon_new_devel.recipe";

# Initial model names
my $init_model = $HMMDIR."/".$BASE_ID;      # Created in tying
my $init_cfg = $HMMDIR."/".$BASE_ID.".cfg"; # Used in tying and training

# Batch settings
my $NUM_BATCHES = 2; # Number of batches, maximum number of parallel jobs
my $BATCH_PRIORITY = 0; # For large batches, use e.g. -100
my $BATCH_MAX_MEM = 2000; # In megabytes

# Train/Baum-Welch settings
my $USE_HMMNETS = 1; # If 0, the script must call align appropriately
my $FORWARD_BEAM = 15;
my $BACKWARD_BEAM = 150;
my $AC_SCALE = 1; # Acoustic scaling (For ML 1, for MMI 1/(LMSCALE/lne(10)))
my $ML_STATS_MODE = "--ml";
my $ML_ESTIMATE_MODE = "--ml";

# Alignment settings
my $ALIGN_WINDOW = 4000;
my $ALIGN_BEAM = 1000;
my $ALIGN_SBEAM = 100;

# Context phone tying options
my $TIE_USE_OUT_PHN = 0; # 0=read transcript field, 1=read alignment field
my $TIE_RULES = "$SCRIPTDIR/finnish_rules.txt";
my $TIE_MIN_COUNT = 1000; # Minimum number of features per state
my $TIE_MIN_GAIN = 3500;  # Minimum loglikelihood gain for a state split
my $TIE_MAX_LOSS = 3500;  # Maximum loglikelihood loss for merging states

# Gaussian splitting options
my $SPLIT_MIN_OCCUPANCY = 300; # Accumulated state probability
my $SPLIT_MAX_GAUSSIANS = 100; # Per state
my $GAUSS_REMOVE_THRESHOLD = 0.001; # Mixture component weight threshold

# Minimum variance
my $MINVAR = 0.1;

# Gaussian clustering options
my $NUM_GAUSS_CLUSTERS = 1000; # 0 if no clustering
my $GAUSS_EVAL_RATIO = 0.1;

# MLLT options
my $mllt_start_iter = 14; # At which iteration MLLT estimation should begin
my $mllt_frequency = 1; # How many EM iterations between MLLT estimation
my $MLLT_MODULE_NAME = "transform";

# Training iterations
my $num_ml_train_iter = 20;
my $split_frequency = 2; # How many EM iterations between Gaussian splits
my $split_stop_iter = 16; # Iteration after which no more splits are done

# Adaptation settings
my $VTLN_MODULE = "vtln";
my $MLLR_MODULE = "mllr";
my $SPKC_FILE = ""; # For initialization see e.g. $SCRIPTDIR/vtln_default.spkc


# Discriminative training settings
my $num_ebw_iter = 4;
my $EBW_STATS_MODE = "--mmi -E";
my $EBW_ESTIMATE_MODE = "--mmi";
my $EBW_AC_SCALE = 0.08;
my $EBW_FORWARD_BEAM = 15;
my $EBW_BACKWARD_BEAM = 20; # Affected by the acoustic scaling 


# Misc settings
my $DUR_SKIP_MODELS = 6; # Models without duration model (silences/noises)
my $VERBOSITY = 1;

# NOTE: if you plan to recompile executables at the same time as running them, 
# it is a good idea to copy the old binaries to a different directory
my $COPY_BINARY_TO_WORK = 0;

my $SAVE_STATISTICS = 1; # Keep the statistics files in iteration directories


######################################################################
# Training script begins
######################################################################

# Create own working directory
my $tempdir = $workdir."/".$BASE_ID;
mkdir $tempdir;
chdir $tempdir || die("Could not chdir to $tempdir");

if ($COPY_BINARY_TO_WORK > 0) {
    copy_binary_to_work($BINDIR, $tempdir."/bin");
}

# Global GridManager object for single processes
my $GM_SINGLE = GridManager->new;
$GM_SINGLE->{"identifier"} = "single";
$GM_SINGLE->{"temp_dir"} = $tempdir;
$GM_SINGLE->{"log_dir"} = $tempdir;
$GM_SINGLE->{"script_dir"} = $SCRIPTDIR;


# Generate initial model by context phone tying using existing alignments
context_phone_tying($init_model, $init_cfg);

# Convert the generated full covariance model to diagonal model
convert_full_to_diagonal($init_model);

# Create the hmmnet files
generate_hmmnet_files($init_model);

# ML/EM training
my $ml_model;
$ml_model=ml_train($tempdir, 1, $num_ml_train_iter, $init_model, $init_cfg,
                   $mllt_start_iter, $mllt_frequency, $split_frequency, $split_stop_iter);
my $om = $ml_model;

# Estimate duration model
align($tempdir, $om, $RECIPE);
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

# Discriminative training
#my $dmodel;
#$AC_SCALE=$DISCRIMINATIVE_AC_SCALE;    # for MMI 1/(LMSCALE/ln(10)))
#$FORWARD_BEAM=$DISCRIMINATIVE_FORWARD_BEAM;
#$BACKWARD_BEAM=$DISCRIMINATIVE_BACKWARD_BEAM;
#$USE_HMMNETS = 1;
#$dmodel=ebw_train($tempdir, 1, $num_ebw_iter, $ml_model, $ml_model.".cfg");
#$om = $dmodel;

# Generate lnas for the final model
generate_lnas($tempdir, $om, $lna_recipe, $lna_outdir);



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

  $GM_SINGLE->submit("$BINDIR/tie -c $im_cfg -o $out_model -r $RECIPE $phn_flag -u $TIE_RULES --count $TIE_MIN_COUNT --sgain $TIE_MIN_GAIN --mloss $TIE_MAX_LOSS -i $VERBOSITY\n");

  if (!(-e $out_model.".ph")) {
    die "Error in context phone tying\n";
  }
}


sub convert_full_to_diagonal {
  my $im = shift(@_);
  my $gk = $im.".gk";
  my $gk_backup = $im."_full.gk";
  
  system("mv $gk $gk_backup");
  system("$BINDIR/gconvert -g $gk_backup -o $gk -d");
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
        print "cp $cur_file.* $log_dir\n";
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


sub ebw_train {
  my $temp_dir = shift(@_);
  my $iter_init = shift(@_);
  my $iter_end = shift(@_);

  my $im = shift(@_);
  my $im_cfg = shift(@_);

  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime(time);
  my $dstring = "$mday.".($mon+1).".".(1900+$year);
  my $model_base = "$HMMDIR/${BASE_ID}_ebw_${dstring}";

  my $stats_list_file = "statsfiles.lst";
  my $batch_info;

  for (my $i = $iter_init; $i <= $iter_end ; $i ++) {

    print "EBW iteration ".$i."\n" if ($VERBOSITY > 0);
    my $om = $model_base."_".$i;

    my $log_dir = "$temp_dir/ebw_iter_".$i;
    mkdir $log_dir;

    collect_stats("ebw_$i", $temp_dir, $log_dir, $im, $im_cfg,
                  $stats_list_file, $EBW_STATS_MODE, 0);

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

    estimate_model($log_dir, $im, $im_cfg, $om, $stats_list_file,
                   $MINVAR, $EBW_ESTIMATE_MODE, 0, 0);
    
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
  my $spkc_switch = "";
  $bw_option = "-H" if ($USE_HMMNETS);
  $spkc_switch = "-S $SPKC_FILE" if ($SPKC_FILE ne "");
  $mllt_option = "--mllt" if ($mllt_flag);

  my $gm = GridManager->new;
  $gm->{"identifier"} = $id;
  $gm->{"run_dir"} = $temp_dir;
  $gm->{"log_dir"} = $log_dir;
  $gm->{"script_dir"} = $SCRIPTDIR;
  if ($NUM_BATCHES > 0) {
    $gm->{"first_batch"} = 1;
    $gm->{"last_batch"} = $NUM_BATCHES;
  }
  $gm->{"priority"} = $BATCH_PRIORITY;
  $gm->{"mem_req"} = $BATCH_MAX_MEM;
  #$gm->{"emergency_shutdown"} = 0;
  $gm->{"failed_batch_retry_count"} = 1;

  $batch_options = "-B $NUM_BATCHES -I \$BATCH" if ($NUM_BATCHES > 0);

  my $failed_batches_file = "${id}_failed_batches.lst";
  unlink($failed_batches_file);

  $gm->submit("$BINDIR/stats -b $model_base -c $cfg -r $RECIPE $bw_option -o temp_stats_\$BATCH $stats_mode -F $FORWARD_BEAM -W $BACKWARD_BEAM -A $AC_SCALE -t $spkc_switch $batch_options -i $VERBOSITY $mllt_option\n".
"if [ \"\$?\" -ne \"0\" ]; then echo \$BATCH >> $failed_batches_file; exit 1; fi\n".
"while ! mkdir stats.lock 2>/dev/null; do sleep 5s; done\n".
"if [ -f ${id}_stats.gks ] ; then\n".
"  echo ${id}_stats > temp_list_\$BATCH\n".
"  echo temp_stats_\$BATCH >> temp_list_\$BATCH\n".
"  sleep 5s\n".
"  $BINDIR/combine_stats -b $model_base -L temp_list_\$BATCH -o ${id}_stats\n".
"  if [ \"\$?\" -ne \"0\" ]; then rmdir stats.lock; exit 1; fi\n".
"  rm temp_list_\$BATCH\n".
"  rm temp_stats_\$BATCH.*\n".
"else\n".
"  mv temp_stats_\$BATCH.gks ${id}_stats.gks\n".
"  mv temp_stats_\$BATCH.mcs ${id}_stats.mcs\n".
"  mv temp_stats_\$BATCH.phs ${id}_stats.phs\n".
"  mv temp_stats_\$BATCH.lls ${id}_stats.lls\n".
"fi\n".
"rmdir stats.lock\n");

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

  my $old_log_dir = $GM_SINGLE->{"log_dir"};
  $GM_SINGLE->{"log_dir"} = $log_dir;

  $extra_options = "--mllt $MLLT_MODULE_NAME" if ($mllt_flag);
  $extra_options = $extra_options." --mremove $GAUSS_REMOVE_THRESHOLD" if ($GAUSS_REMOVE_THRESHOLD > 0);
  $extra_options = $extra_options." --split $SPLIT_MIN_OCCUPANCY --maxg $SPLIT_MAX_GAUSSIANS" if ($split_flag);

  $GM_SINGLE->submit("$BINDIR/estimate -b $im -c $im_cfg -L $stats_list_file -o $om -t -i $VERBOSITY --minvar $minvar $estimate_mode -s ${BASE_ID}_summary $extra_options\n");
  $GM_SINGLE->{"log_dir"} = $old_log_dir;
}


sub generate_hmmnet_files {
  my $im = shift(@_);
  $GM_SINGLE->submit("export PERL5LIB=$SCRIPTDIR\n$SCRIPTDIR/make_hmmnets.pl $im.ph $RECIPE");
}


sub align {
  my $temp_dir = shift(@_);
  my $model = shift(@_);
  my $recipe = shift(@_);
  my $spkc_file = shift(@_);
  my $batch_options = "";
  my $spkc_switch = "";
  $spkc_switch = "-S $spkc_file" if ($spkc_file ne "");

  my $gm = GridManager->new;
  $gm->{"identifier"} = "align_${BASE_ID}";
  $gm->{"run_dir"} = $temp_dir;
  $gm->{"log_dir"} = $temp_dir;
  $gm->{"script_dir"} = $SCRIPTDIR;
  if ($NUM_BATCHES > 0) {
    $gm->{"first_batch"} = 1;
    $gm->{"last_batch"} = $NUM_BATCHES;
  }
  $gm->{"priority"} = $BATCH_PRIORITY;
  #$gm->{"mem_req"} = $BATCH_MAX_MEM; # No large mem requirements
  #$gm->{"emergency_shutdown"} = 0;
  $gm->{"failed_batch_retry_count"} = 1;

  $batch_options = "-B $NUM_BATCHES -I \$BATCH" if ($NUM_BATCHES > 0);
  $gm->submit("$BINDIR/align -b $model -c $model.cfg -r $recipe --swins $ALIGN_WINDOW --beam $ALIGN_BEAM --sbeam $ALIGN_SBEAM $spkc_switch $batch_options -i $VERBOSITY\n");
}


# This version runs Baum-Welch
sub estimate_mllr {
  my $temp_dir = shift(@_);
  my $model = shift(@_);
  my $recipe = shift(@_);
  my $out_file = shift(@_);
  my $temp_out;
  my $batch_options = "";

  my $gm = GridManager->new;
  $gm->{"identifier"} = "mllr_${BASE_ID}";
  $gm->{"run_dir"} = $temp_dir;
  $gm->{"log_dir"} = $temp_dir;
  $gm->{"script_dir"} = $SCRIPTDIR;
  if ($NUM_BATCHES > 0) {
    $gm->{"first_batch"} = 1;
    $gm->{"last_batch"} = $NUM_BATCHES;
  }
  $gm->{"priority"} = $BATCH_PRIORITY;
  $gm->{"mem_req"} = $BATCH_MAX_MEM; # No large mem requirements
  #$gm->{"emergency_shutdown"} = 0;
  $gm->{"failed_batch_retry_count"} = 1;

  $temp_out = $out_file;
  if ($NUM_BATCHES > 1) {
    $batch_options = "-B $NUM_BATCHES -I \$BATCH";
    $temp_out = "mllr_temp_\$BATCH.spkc";
  }
  $gm->submit("$BINDIR/mllr -b $model -c $model.cfg -r $recipe -H -F $FORWARD_BEAM -W $BACKWARD_BEAM -M $MLLR_MODULE -S $SPKC_FILE -o $temp_out $batch_options -i $VERBOSITY");

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

  my $gm = GridManager->new;
  $gm->{"identifier"} = "vtln_${BASE_ID}.sh";
  $gm->{"run_dir"} = $temp_dir;
  $gm->{"log_dir"} = $temp_dir;
  $gm->{"script_dir"} = $SCRIPTDIR;
  if ($NUM_BATCHES > 0) {
    $gm->{"first_batch"} = 1;
    $gm->{"last_batch"} = $NUM_BATCHES;
  }
  $gm->{"priority"} = $BATCH_PRIORITY;
  #$gm->{"mem_req"} = $BATCH_MAX_MEM; # No large mem requirements
  #$gm->{"emergency_shutdown"} = 0;
  $gm->{"failed_batch_retry_count"} = 1;

  $temp_out = $out_file;
  if ($NUM_BATCHES > 1) {
    $batch_options = "-B $NUM_BATCHES -I \$BATCH";
    $temp_out = "vtln_temp_\$BATCH.spkc";
  }
  $gm->submit("$BINDIR/vtln -b $model -c $model.cfg -r $recipe -O -v $VTLN_MODULE -S $SPKC_FILE -o $temp_out $batch_options -i $VERBOSITY");
  if ($NUM_BATCHES > 1) {
    system("cat vtln_temp_*.spkc > $out_file") && die("vtln estimation failed");
  }
}



# NOTE: Uses alignments
sub estimate_dur_model {
  my $om = shift(@_);
  $GM_SINGLE->submit("$BINDIR/dur_est -p $om.ph -r $RECIPE -O --gamma $om.dur --skip $DUR_SKIP_MODELS");
  if (!(-e $om.".dur")) {
    die("Error estimating duration models\n");
  }
}


sub cluster_gaussians {
  my $im = shift(@_);
  $GM_SINGLE->submit("$BINDIR/gcluster -g $im.gk -o $im.gcl -C $NUM_GAUSS_CLUSTERS -i $VERBOSITY");
  if (!(-e $im.".gcl")) {
    die("Error in Gaussian clustering\n");
  }
}



sub generate_lnas {
  my $temp_dir = shift(@_);
  my $model = shift(@_);
  my $recipe = shift(@_);
  my $out_dir = shift(@_);
  my $spkc_file = shift(@_);
  my $batch_options = "";
  my $cluster_options = "";

  my $spkc_switch = "";
  $spkc_switch = "-S $spkc_file" if ($spkc_file ne "");

  mkdir $out_dir;

  my $gm = GridManager->new;
  $gm->{"identifier"} = "lna_${BASE_ID}";
  $gm->{"run_dir"} = $temp_dir;
  $gm->{"log_dir"} = $temp_dir;
  $gm->{"script_dir"} = $SCRIPTDIR;
  if ($NUM_BATCHES > 0) {
    $gm->{"first_batch"} = 1;
    $gm->{"last_batch"} = $NUM_BATCHES;
  }
  $gm->{"priority"} = $BATCH_PRIORITY;
  #$gm->{"mem_req"} = $BATCH_MAX_MEM; # No large mem requirements
  #$gm->{"emergency_shutdown"} = 0;
  $gm->{"failed_batch_retry_count"} = 1;

  $batch_options = "-B $NUM_BATCHES -I \$BATCH" if ($NUM_BATCHES > 0);
  $cluster_options = "-C $model.gcl --eval-ming $GAUSS_EVAL_RATIO" if ($NUM_GAUSS_CLUSTERS);
  $gm->submit("$BINDIR/phone_probs -b $model -c $model.cfg -r $recipe -o $out_dir $spkc_switch $batch_options -i $VERBOSITY $cluster_options");
}
