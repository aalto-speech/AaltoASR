#!/usr/bin/perl

# This script performs discriminative training using the Extended Baum-Welch
# algorithm. Supported discriminative criteria: MMI, MPE, MWE, MPFE.
# Prerequisites:
#   - Trained ML model, which is used to initialize the discriminative training
#   - Denominator lattices (hmmnets) generated with generate_den_hmmnets.pl,
#     using the ML model (in fact, the model for generating denominator
#     lattices may be different, as long as the model state topology
#     (context phone tying) is the same)

# This is a TEMPLATE SCRIPT, you need to modify the paths, input files
# and options according to your needs!


use lib '/share/puhe/scripts/cluster'; # For CondorManager
use locale;
use strict;
use CondorManager;


## Model name ##
  my $BASE_ID="speecon_mmi";

## Path settings ##
  my $BINDIR="/home/".$ENV{"USER"}."/aku/cvs/aku";
  my $HMMDIR="/share/puhe/".$ENV{"USER"}."/hmms";
  my $workdir="/share/work/".$ENV{"USER"}."/aku_work";

## Training file list ##
  my $RECIPE="/full/path/of/your.recipe";

## Initial model names (Fully trained ML model) ##
  my $init_model = "$HMMDIR/speecon_ml_20";
  my $init_cfg = $init_model.".cfg";

## Batch settings ##
  my $NUM_BATCHES = 50; # Number of batches, maximum number of parallel jobs
  my $BATCH_PRIORITY = 0; # Not used with Condor
  my $BATCH_MAX_MEM = 3000; # In megabytes. Not used with Condor.

## Baum-Welch settings ##
  my $FORWARD_BEAM = 15;
  my $BACKWARD_BEAM = 40; # Depends on AC_SCALE
  my $AC_SCALE = 0.077; # Acoustic scaling (1/(LMSCALE/lne(10)))

## Extended Baum-Welch settings ##
  # Statistics collection
  # It is usually best to use segmentation mode Multipath Viterbi (-M mpv),
  # although plain Baum-Welch (-M bw) might work well for MMI. With current
  # implementation, using BW-segmentation with MPE is very slow!
  # Examples:
  # MMI training: -M mpv --mmi
  # MPE training: -M mpv --mpe --errmode=mpe
  # For MPFE, the following is recommended: -M mpv --mpe --errmode=mpfe --nosil=<w>
  #     NOTE! Replace <w> with the proper silence label (e.g. _ in english models)
  my $STATS_MODE = "-M mpv --mmi";

  # Model estimation options. The discriminative criterion needs to match
  # with the statistics collection. The widely used defaults for EBW estimate
  # options are:
  # MMI: --mmi --ismooth=0
  # MPE: --mpe --ismooth=50
  # MPFE: --mpe --ismooth=400
  # However, these are not optimal in all situations, e.g. when the number
  # of features vs. Gaussians is outside the usual range.
  # To produce robust models, it is recommended to tweak the so called D
  # constant: --C1=0 --ismooth=D --prev-prior
  # Determine D so that the median KLD change of the Gaussians in the
  # first EBW update (from the ML model) is reasonable,
  # e.g. 0.01 for MMI and 0.1 for MPE/MPFE. This can be computed with
  # clskld -g
  my $ESTIMATE_MODE = "--mmi --C1=0 --ismooth=300 --prev-prior";


## Minimum variance ##
  my $MINVAR = 0.10;

## Number of iterations ##
  # It is important to test the resulting models from multiple iterations
  # using a development test set. The results of MMI models can exhibit
  # significant oscillation, and MPE/MPFE models may suffer from over-learning.
  my $num_ebw_iter = 8;

## Adaptation settings ##
  my $SPKC_FILE = "";

## Misc settings ##
  my $VERBOSITY = 1;

  # NOTE: if you plan to recompile executables at the same time as running
  # them, it is a good idea to copy the old binaries to a different directory
  my $copy_binary_to_work = 1;

  my $SAVE_STATISTICS = 1;


######################################################################
# Training script begins
######################################################################

# Create own working directory
my $tempdir = $workdir."/".$BASE_ID;
mkdir $tempdir;
chdir $tempdir || die("Could not chdir to $tempdir");

if ($copy_binary_to_work > 0) {
    copy_binaries($BINDIR, $tempdir."/bin");
}

my $om;
$om=ebw_train($tempdir, 1, $num_ebw_iter, $init_model, $init_cfg);

print "New model $om\n";


sub copy_binaries {
    my $orig_bin_dir = shift(@_);
    my $new_bin_dir = shift(@_);

    mkdir $new_bin_dir;
    system("cp ${orig_bin_dir}/stats ${new_bin_dir}/stats");
    system("cp ${orig_bin_dir}/estimate ${new_bin_dir}/estimate");
    system("cp ${orig_bin_dir}/gcluster ${new_bin_dir}/gcluster");
    system("cp ${orig_bin_dir}/combine_stats ${new_bin_dir}/combine_stats");
    $BINDIR = $new_bin_dir;
}



sub ebw_train {
  my $temp_dir = shift(@_);
  my $iter_init = shift(@_);
  my $iter_end = shift(@_);

  my $im = shift(@_);
  my $cfg = shift(@_);

  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime(time);
  my $dstring = "$mday.".($mon+1).".".(1900+$year);
  my $model_base = "$HMMDIR/${BASE_ID}_${dstring}";

  my $stats_list_file = "statsfiles.lst";
  my $batch_info;
  my @stats_files;

  for (my $i = $iter_init; $i <= $iter_end ; $i ++) {

    print "Iteration ".$i."\n" if ($VERBOSITY > 0);
    my $om = $model_base."_".$i;

    my $log_dir = "$temp_dir/iter_".$i;
    mkdir $log_dir;

    collect_stats("genstats_$i", $temp_dir, $log_dir, $im, $cfg, 
                  $stats_list_file, $STATS_MODE);

    my $fh;
    open($fh, $stats_list_file) || die("Could not open file $stats_list_file!");
    @stats_files=<$fh>;
    close($fh);

    if ($SAVE_STATISTICS) {
      my $cur_file;
      foreach $cur_file (@stats_files) {
        chomp $cur_file;
        system("cp $cur_file.* $log_dir");
      }
      system("cp $stats_list_file $log_dir");
    }

    estimate_model($log_dir, $im, $cfg, $om, $stats_list_file, $MINVAR);
    
    # Check the models were really created
    if (!(-e $om.".ph")) {
      # Give NFS some time
      sleep(15);
      if (!(-e $om.".ph")) {
        die "Model optimization terminated\n";
      }
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
  }
  system("rm stats_*.*"); # Remove statistics files

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
  my $spkc_switch = "";
  my $mllt_option = "";
  $spkc_switch = "-S $SPKC_FILE" if ($SPKC_FILE ne "");

  my $cm = CondorManager->new;
  $cm->{"identifier"} = $id;
  $cm->{"run_dir"} = $temp_dir;
  $cm->{"log_dir"} = $log_dir;
  if ($NUM_BATCHES > 0) {
    $cm->{"first_batch"} = 1;
    $cm->{"last_batch"} = $NUM_BATCHES;
  }
  $cm->{"priority"} = $BATCH_PRIORITY;
  $cm->{"mem_req"} = $BATCH_MAX_MEM;
  $cm->{"failed_batch_retry_count"} = 1;

  my $batch_options = "-B $NUM_BATCHES -I \$BATCH" if ($NUM_BATCHES > 0);

  my $failed_batches_file = "${id}_failed_batches.lst";
  unlink($failed_batches_file);

  $cm->submit("$BINDIR/stats -b $model_base -c $cfg -r $RECIPE -H -o temp_stats_\$BATCH $stats_mode -F $FORWARD_BEAM -W $BACKWARD_BEAM -A $AC_SCALE $spkc_switch $batch_options -i $VERBOSITY\n".
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

  system("$BINDIR/estimate -b $im -c $im_cfg -L $stats_list_file -o $om -i $VERBOSITY --minvar $minvar $ESTIMATE_MODE -s ${BASE_ID}_summary > $log_dir/estimate.stdout 2> $log_dir/estimate.stderr\n");
}
