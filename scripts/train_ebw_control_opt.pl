#!/usr/bin/perl

# This is an example script showing how to use opt_ebw_d to control EBW training.
# It shares many parts with train_ebw.pl
# Prerequisites:
#   - Trained ML model, which is used to initialize the discriminative training
#   - Denominator lattices (hmmnets) generated with generate_den_hmmnets.pl,
#     using the ML model (in fact, the model for generating denominator
#     lattices may be different, as long as the model state topology (tying)
#     is the same)

use lib '/home/jpylkkon/lib';
use locale;
use strict;
use ClusterManager;



## Model name ##
  my $BASE_ID="speecon_controlled_ebw_mmi";

## Path settings ##
  my $BINDIR="/home/".$ENV{"USER"}."/aku/cvs/aku";
  my $HMMDIR="/share/puhe/".$ENV{"USER"}."/hmms";
  my $workdir="/share/work/".$ENV{"USER"}."/aku_work";


## Training file list ##
  my $TRAIN_RECIPE="/full/path/of/your_train.recipe";

  # Control recipe is a held-out set used for optimizing EBW constants. The
  # statistics are included for the final update.
  # Reasonable size: e.g. about 10% of the $TRAIN_RECIPE
  # Two versions are needed:
  # $TRAIN_CONTROL_RECIPE: Contains lattices with weak language model, for
  # computing training statistics
  # $CONTROL_RECIPE: Contains lattices with full language model, for computing
  # SNFE measures used in control
  my $TRAIN_CONTROL_RECIPE="/full/path/of/your_train_control.recipe";
  my $CONTROL_RECIPE="/full/path/of/your_control.recipe";

## Initial model names (Fully trained ML model) ##
  my $init_model = "$HMMDIR/speecon_ml_gain3500_occ300_21.7.2011_22";
  my $init_cfg = $init_model.".cfg";

## Batch settings ##
  my $NUM_BATCHES = 50; # Number of batches, maximum number of parallel jobs
  my $BATCH_PRIORITY = 0; # Increase if you want to be nice
  my $BATCH_MAX_MEM = 3500; # In megabytes

## Baum-Welch settings ##
  my $FORWARD_BEAM = 15;
  my $BACKWARD_BEAM = 40;
  my $AC_SCALE = 0.077; # Acoustic scaling (1/(LMSCALE/lne(10)))

## Extended Baum-Welch settings ##
  # Statistics collection
  my $TRAIN_STATS_MODE = "-M mpv --mmi";
  my $CONTROL_STATS_MODE = "-M mpv --mpe --errmode=snfe";
  my $ESTIMATE_MODE = "--mmi --C1=0";


## EBW control parameters ##
  my $INITIAL_D = 392;  # Change this! (e.g. to median of the normal D values)
  my $MAX_D_MULT = 2;   # Range of control parameter change: maximum multiplier
  my $MIN_D_MULT = 0.5; # Range of control parameter change: minimum multiplier
  my $MAX_CONTROL_ITER = 30;

  # --gmin needs to match with estimate_model()-call minimum D (--C2), in this case
  # 2*0.75=1.5 (because initial estimate_model uses --C2=2)
  # D constant clustering can be defined with --cluster=global/pho/mix
  my $CONTROL_MODE = "--control=mpe --train=mmi --qp-eps=0.02 --prior=5 --gmin=0.75";
  my $D_OPT_INITIAL_STEP = 0.02;

# NOTE! You must provide initial file for (possibly clustered) D constants.
# File name is "$tempdir/${BASE_ID}_0.ebwd"
# Each line contains: D D_min D_max
#  which can be the same for all the lines. E.g.: 200 100 400
# Number of lines: --cluster=global -> 1
#                  --cluster=pho    -> Number of phones
#                  --cluster=mix    -> Number of mixtures
#                  <NONE>           -> Number of Gaussians


## Minimum variance ##
  my $MINVAR = 0.10;

## Number of iterations ##
  # It is important to test the resulting models from multiple iterations
  # using a development test set. The results of MMI models can exhibit
  # significant oscillation, and MPE/MPFE models may suffer from over-learning.
  my $num_ebw_iter = 15;

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
$om=ebw_control_train($tempdir, 1, $num_ebw_iter, $init_model, $init_cfg);

print "New model $om\n";


sub copy_binaries {
    my $orig_bin_dir = shift(@_);
    my $new_bin_dir = shift(@_);

    mkdir $new_bin_dir;
    system("cp ${orig_bin_dir}/stats ${new_bin_dir}/stats");
    system("cp ${orig_bin_dir}/estimate ${new_bin_dir}/estimate");
    system("cp ${orig_bin_dir}/gcluster ${new_bin_dir}/gcluster");
    system("cp ${orig_bin_dir}/combine_stats ${new_bin_dir}/combine_stats");
    system("cp ${orig_bin_dir}/opt_ebw_d ${new_bin_dir}/opt_ebw_d");
    $BINDIR = $new_bin_dir;
}



sub ebw_control_train {
  my $temp_dir = shift(@_);
  my $iter_init = shift(@_);
  my $iter_end = shift(@_);

  my $im = shift(@_);
  my $cfg = shift(@_);

  my ($sec,$min,$hour,$mday,$mon,$year,$wday,$yday,$isdst) = localtime(time);
  my $dstring = "$mday.".($mon+1).".".(1900+$year);
  my $model_base = "$HMMDIR/${BASE_ID}_${dstring}";

  my $train_stats_list_file = "statsfiles.lst";
  my $train_control_stats_list_file = "statsfiles_extra.lst";
  my $batch_info;
  my @stats_files;
  my @stats_extra_files;

  for (my $i = $iter_init; $i <= $iter_end ; $i ++) {

    print "Iteration ".$i."\n" if ($VERBOSITY > 0);
    my $om = $model_base."_".$i;
    my $cur_ebwd = "${BASE_ID}_".$i.".ebwd";

    my $log_dir = "$temp_dir/iter_".$i;
    mkdir $log_dir;

    #system("rm stats_*.*"); # Delete previous statistics

    my $cur_train_recipe = $TRAIN_RECIPE;
    my $cur_train_control_recipe = $TRAIN_CONTROL_RECIPE;
    my $cur_control_recipe = $CONTROL_RECIPE;

    collect_stats("genstats_$i", $NUM_BATCHES, $temp_dir, $log_dir,
                  $im, $cfg, $train_stats_list_file,
                  $TRAIN_STATS_MODE, $cur_train_recipe);
    collect_stats("genstats_ctrl_$i", $NUM_BATCHES/2, $temp_dir, $log_dir,
                  $im, $cfg, $train_control_stats_list_file,
                  $TRAIN_STATS_MODE, $cur_train_control_recipe);
    my $fh;
    open($fh, $train_stats_list_file) || die("Could not open file $train_stats_list_file!");
    @stats_files=<$fh>;
    close($fh);
    open($fh, $train_control_stats_list_file) || die("Could not open file $train_control_stats_list_file!");
    @stats_extra_files=<$fh>;
    close($fh);


    if ($SAVE_STATISTICS) {
      my $cur_file;
      foreach $cur_file (@stats_files) {
        chomp $cur_file;
        system("cp $cur_file.* $log_dir");
      }
      foreach $cur_file (@stats_extra_files) {
        chomp $cur_file;
        system("cp $cur_file.* $log_dir");
      }
      system("cp $train_stats_list_file $log_dir");
      system("cp $train_control_stats_list_file $log_dir");
    }

    # Copy the previous iteration D file as initialization
    system("cp ${BASE_ID}_".($i-1).".ebwd $cur_ebwd");

    control_loop($cur_control_recipe, $temp_dir, $log_dir, $im, $cfg, $i, $om,
                 $train_stats_list_file, $train_control_stats_list_file,
                 $cur_ebwd);
    
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
    foreach $cur_file (@stats_extra_files) {
      chomp $cur_file;
      system("rm $cur_file.*");
    }
    #system("rm $stats_list_file");

    # Read input from previously written model
    $im = $om;
  }
  # system("rm stats_*.*"); # Remove statistics files

  return $im;
}


sub control_loop {
  my $control_recipe = shift(@_);
  my $temp_dir = shift(@_);
  my $log_dir = shift(@_);
  my $train_model = shift(@_);
  my $cfg = shift(@_);
  my $trit = shift(@_);
  my $om = shift(@_);
  my $stats_list_file = shift(@_);
  my $held_out_stats_list_file = shift(@_);
  my $cur_d_file = shift(@_);
  my $out_d_file;
  my $gaussian_d_file;
  my @cur_control_stats_files;

  my @d_array;

  my $control_stats_list_file = "control_statsfiles.lst";
  
  # Create segmented lattices of the control set
  collect_stats("control_${trit}_savelat", $NUM_BATCHES, $temp_dir, $log_dir,
                $train_model, $cfg, $control_stats_list_file,
                $CONTROL_STATS_MODE." --savelat", $control_recipe);

  # Collect the current median of the D-values
  my $fh_in;
  my @d_values;
  open($fh_in, "< $cur_d_file") || die("Could not open file $cur_d_file!");
  while (<$fh_in>) {
    my @c = split;
    push @d_values, $c[0];
  }
  close($fh_in);
  my $mean_d_value = mean(@d_values);

  # Set clustered D limits
  # Update D/min D/max D 
  my $fh_out;
  open($fh_in, "< $cur_d_file") || die("Could not open file $cur_d_file!");
  open($fh_out, "> ${cur_d_file}.new") || die("Could not open file $cur_d_file!");
  while (<$fh_in>) {
    my @c = split;
    my $d = $c[0];
    my $dmin = $mean_d_value * $MIN_D_MULT;
    my $dmax = $mean_d_value * $MAX_D_MULT;
    if ($d == 0) {
      $dmin = 0;
      $dmax = 0;
    } else {
      $d = $dmin if ($d > 0 && $d < $dmin);
      $d = $dmax if ($dmax > 0 && $d > $dmax);
    }
    print $fh_out $d." ".$dmin." ".$dmax."\n";
  }
  close($fh_out);
  close($fh_in);
  system("mv ${cur_d_file}.new $cur_d_file");

  # Remove a possible state file
  unlink("control.state");

  # Initialize Gaussian D file
  $gaussian_d_file = $cur_d_file.".gaussian";
  my $ret = system("$BINDIR/opt_ebw_d -b $train_model -L $control_stats_list_file -T $stats_list_file -F control.state --cluster-d $cur_d_file -o $gaussian_d_file $CONTROL_MODE --d-init -A $AC_SCALE -i 2 > $log_dir/opt_ebw_${trit}_0.stdout 2> $log_dir/opt_ebw_${trit}_0.stderr");
  die "Error running opt_ebw_d" if ($ret == -1 || ($ret>>8) > 0);

  my $control_model = "${BASE_ID}_${trit}_control";

  # Set Gaussian D files to initial values
  estimate_model($log_dir, $train_model, $cfg, $control_model,
                 $stats_list_file, 2, $MINVAR, 0,
                 $gaussian_d_file, $gaussian_d_file);

  # Re-run opt_ebw_d to count the new Gaussian D limits
  my $ret = system("$BINDIR/opt_ebw_d -b $train_model -L $control_stats_list_file -T $stats_list_file -F control.state -D $gaussian_d_file --cluster-d $cur_d_file -o $gaussian_d_file $CONTROL_MODE --d-init -A $AC_SCALE -i 2 > $log_dir/opt_ebw_${trit}_0.stdout 2> $log_dir/opt_ebw_${trit}_0.stderr");
  die "Error running opt_ebw_d" if ($ret == -1 || ($ret>>8) > 0);

  # DEBUG!!!
  #system("cp $gaussian_d_file ${gaussian_d_file}.init");

  for (my $ci = 1; $ci <= $MAX_CONTROL_ITER; $ci++) {
    
    # Store D-files for later usage
    my $this_it_d_file = ${cur_d_file}.".".$ci;
    my $this_it_gd_file = ${gaussian_d_file}.".".$ci;
    system("cp $cur_d_file $this_it_d_file");
    system("cp $gaussian_d_file $this_it_gd_file");

    # Estimate a model with previous D-values
    estimate_model($log_dir, $train_model, $cfg, $control_model,
                   $stats_list_file, 1.5, $MINVAR, 0,
                   $gaussian_d_file);
    
    # Collect control set statistics
    collect_stats("control_${trit}_$ci", $NUM_BATCHES/2,
                  $temp_dir, $log_dir, $control_model, $cfg,
                  $control_stats_list_file,
                  $CONTROL_STATS_MODE." -P", $control_recipe);

    # Using the training set and the development set statistics, optimize
    # the D values based on the control criterion
    my $init_switch = "-l ".$D_OPT_INITIAL_STEP;
    $init_switch = "" if ($ci > 1);
    my $ret = system("$BINDIR/opt_ebw_d -b $train_model -L $control_stats_list_file -T $stats_list_file -F control.state -D $gaussian_d_file -o $gaussian_d_file --cluster-d $cur_d_file $CONTROL_MODE $init_switch -A $AC_SCALE -i 2 > $log_dir/opt_ebw_${trit}_${ci}.stdout 2> $log_dir/opt_ebw_${trit}_${ci}.stderr");
    die "Error running opt_ebw_d" if ($ret == -1);

    # Store the model performance from the .lls file to evaluate the
    # models after the control iterations
    my $temp_file = "$log_dir/opt_ebw_${trit}_${ci}.stderr";
    my $score = `grep \"score\" $temp_file | cut -f 2 -d ':'`;
    chomp $score;
    push @d_array, {score => $score, ebwd => $this_it_d_file,
                    gd => $this_it_gd_file };
    print "Iteration $ci: SCORE ".$score."\n";

    # Delete the control statistics
    my $fh;
    open($fh, $control_stats_list_file) || die("Could not open file $control_stats_list_file!");
    @cur_control_stats_files=<$fh>;
    close($fh);

    my $cur_file;
    foreach $cur_file (@cur_control_stats_files) {
      chomp $cur_file;
      system("rm $cur_file.*");
    }

    if (($ret>>8) > 0) {
      print "Optimization terminated (".($ret>>8).")\n";
      last;
    }
    
    # Compare the D files, end optimization if no changes occurred
    my $diff = `diff $this_it_d_file $cur_d_file`;
    chomp $diff;
    last if (length($diff) == 0);
  }

  # Pick the best model and D values (for initializing the next iteration).
  # Copy the best model as the output model and remove the temporary models.
  my $best_iter = 0;
  for (my $i = 1; $i <= $#d_array; $i++) {
    if ($d_array[$i]{score} < $d_array[$best_iter]{score}) {
      $best_iter = $i;
    }
  }
  print "Best model iteration ".($best_iter+1).": SCORE ".
      $d_array[$best_iter]{score}."\n";
  my $fh;
  if (open($fh, ">> ${BASE_ID}_control_summary")) {
    print $fh "Best model iteration ".($best_iter+1).": SCORE ".
        $d_array[$best_iter]{score}."\n";
    close($fh);
  }
  system("cp ".$d_array[$best_iter]{ebwd}." $cur_d_file");
  system("cp ".$d_array[$best_iter]{gd}." $gaussian_d_file");
  system("rm ".$control_model.".*");
  for my $href ( @d_array ) {
    system("rm ".$href->{ebwd});
    system("rm ".$href->{gd});
  }

  # Estimate the final model, include the held-out set to the statistics
  system("cat $stats_list_file $held_out_stats_list_file > $control_stats_list_file");
  estimate_model($log_dir, $train_model, $cfg, $om,
                 $control_stats_list_file, 1.5, $MINVAR, 1,
                 $gaussian_d_file);
}


sub collect_stats {
  my $id = shift(@_);
  my $num_batches = shift(@_);
  my $temp_dir = shift(@_);
  my $log_dir = shift(@_);
  my $model_base = shift(@_);
  my $cfg = shift(@_);
  my $stats_list_file = shift(@_);
  my $stats_mode = shift(@_);
  my $recipe = shift(@_);
  my $spkc_switch = "";
  my $mllt_option = "";
  $spkc_switch = "-S $SPKC_FILE" if ($SPKC_FILE ne "");

  my $cm = ClusterManager->new;
  $cm->{"identifier"} = $id;
  $cm->{"run_dir"} = $temp_dir;
  $cm->{"log_dir"} = $log_dir;
  $cm->{"run_time"} = 239;
  if ($num_batches > 0) {
    $cm->{"first_batch"} = 1;
    $cm->{"last_batch"} = $num_batches;
  }
  $cm->{"priority"} = $BATCH_PRIORITY;
  $cm->{"mem_req"} = $BATCH_MAX_MEM;
  $cm->{"failed_batch_retry_count"} = 0;

  my $batch_options = "-B $num_batches -I \$BATCH" if ($num_batches > 0);

  my $failed_batches_file = "${id}_failed_batches.lst";
  unlink($failed_batches_file);

  $cm->submit("$BINDIR/stats -b $model_base -c $cfg -r $recipe -H -o temp_stats_\$BATCH $stats_mode -F $FORWARD_BEAM -W $BACKWARD_BEAM -A $AC_SCALE $spkc_switch $batch_options -i $VERBOSITY\n".
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
"  if [ -e temp_stats_\$BATCH.lls ] ; then\n".
"    mv temp_stats_\$BATCH.gks ${id}_stats.gks\n".
"    mv temp_stats_\$BATCH.mcs ${id}_stats.mcs\n".
"    mv temp_stats_\$BATCH.phs ${id}_stats.phs\n".
"    mv temp_stats_\$BATCH.lls ${id}_stats.lls\n".
"  fi\n".
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
  my $mind_constant = shift(@_);
  my $minvar = shift(@_);
  my $save_summary = shift(@_);
  my $in_ebwd = shift(@_);
  my $out_ebwd = shift(@_);
  #my $mixtures = shift(@_);
  my $extra_switches = "";

  $extra_switches = $extra_switches."-D $in_ebwd " if (defined $in_ebwd);
  $extra_switches = $extra_switches."--write-ebwd=$out_ebwd " if (defined $out_ebwd && length($out_ebwd) > 0);
  $extra_switches = $extra_switches."-s ${BASE_ID}_summary " if ($save_summary);
  #$extra_switches = $extra_switches."--no-mixture-update" if (defined $mixtures && $mixtures == 0);

  system("$BINDIR/estimate -b $im -c $im_cfg -L $stats_list_file -o $om -i $VERBOSITY --C2=$mind_constant --minvar $minvar $ESTIMATE_MODE $extra_switches > $log_dir/estimate.stdout 2> $log_dir/estimate.stderr\n");
}


# From StatLite.pm:
sub median
{
  return unless @_;
  return $_[0] unless @_ > 1;
  @_= sort{$a<=>$b}@_;
  return $_[$#_/2] if @_&1;
  my $mid= @_/2;
  return ($_[$mid-1]+$_[$mid])/2;
}

sub sum
{
	return unless @_;
	return $_[0] unless @_ > 1;
	my $sum;
	foreach(@_) { $sum+= $_; }
	return $sum;
}

sub mean
{
  return unless @_;
  return $_[0] unless @_ > 1;
  return sum(@_)/scalar(@_);
}
