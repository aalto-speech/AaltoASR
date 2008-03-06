#!/usr/bin/perl

# DO NOT USE THIS SCRIPT DIRECTLY! Copy it to your own directory and
# modify appropriately. You MUST modify at least the path settings,
# training file list and BASE_ID/initial model names!

# Run this script at itl-cl1, as it uses GridEngine for scheduling
# the parallel processes.

use locale;
use strict;

# Model name
my $BASE_ID="mpe";

# Path settings
my $BINDIR="/home/".$ENV{"USER"}."/aku";
my $SCRIPTDIR="$BINDIR/scripts";
my $HMMDIR="/share/puhe/".$ENV{"USER"}."/hmms";
my $workdir="/share/work/".$ENV{"USER"}."/aku_work";

# Training file list
my $RECIPE="/share/puhe/jpylkkon/speecon_new/speecon_new_train.recipe";

# Initial model names
my $init_model = $HMMDIR."/ml_model";
my $init_cfg = $init_model.".cfg";

# Batch settings
my $NUM_BATCHES = 4; # Number of processes in parallel

# Train/Baum-Welch settings
my $FORWARD_BEAM = 20;
my $BACKWARD_BEAM = 200;
my $AC_SCALE = 0.075; # Acoustic scaling (1/(LMSCALE/lne(10)))
my $STATS_MODE = "-E --mpegrad --mpemode=hyp-cp-state";

# Minimum variance
my $MINVAR = 0.09;

# Optimization settings
my $num_opt_iter = 6;
my $NUM_BFGS_UPDATES = 4;
my $INIT_INV_HESSIAN = 200;

# Adaptation settings
my $SPKC_FILE = "";

# Misc settings
my $VERBOSITY = 1;

# NOTE: if you plan to recompile executables at the same time as running them, 
# it is a good idea to copy the old binaries to a different directory
my $copy_binary_to_work = 0;

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
$om=mpe_optimize($tempdir, 1, $num_opt_iter, $init_model, $init_cfg);

print "New model $om\n";


sub copy_binaries {
    my $orig_bin_dir = shift(@_);
    my $new_bin_dir = shift(@_);

    mkdir $new_bin_dir;
    system("cp ${orig_bin_dir}/stats ${new_bin_dir}/stats");
    system("cp ${orig_bin_dir}/optmodel ${new_bin_dir}/optmodel");
    $BINDIR = $new_bin_dir;
}


sub mpe_optimize {
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
  my $init_options;

  for (my $i = $iter_init; $i <= $iter_end ; $i ++) {

    print "Iteration ".$i."\n" if ($VERBOSITY > 0);
    my $om = $model_base."_".$i;

    collect_stats($temp_dir, $im, $cfg, $stats_list_file, 0);
    if ($i == 1) {
      $init_options = "-l $INIT_INV_HESSIAN";
    } else {
      $init_options = "";
    }
    optimization_step($temp_dir, $im, $om, $stats_list_file,
                      $init_options);
    
    # Check the models were really created
    if (!(-e $om.".ph")) {
      die "MPE optimization terminated\n";
    }

    # Read input from previously written model
    $im = $om;
  }
  return $im;
}


sub collect_stats {
  my $temp_dir = shift(@_);
  my $model_base = shift(@_);
  my $cfg = shift(@_);
  my $stats_list_file = shift(@_);
  my $batch_options;
  my ($scriptfile, $statsfile, $keyfile);
  my $fh;
  my $batch_info = get_empty_batch_info();
  my $list_fh;
  my $spkc_switch = "";
  $spkc_switch = "-S $SPKC_FILE" if ($SPKC_FILE ne "");

  open $list_fh, "> $stats_list_file" || die "Could not open $stats_list_file";

  $scriptfile = "genstats_${BASE_ID}.sh";
  open $fh, "> $scriptfile" || die "Could not open $scriptfile";
  $statsfile = "stats";
  $keyfile = "stats_ready";
  $batch_options = get_aku_batch_options($NUM_BATCHES, $batch_info);
  if ($NUM_BATCHES > 1) {
    for (my $i = 1; $i <= $NUM_BATCHES; $i++) {
      my $cur_keyfile = $keyfile."_$i";
      my $cur_statsfile = $statsfile."_$i";
      print $list_fh $cur_statsfile."\n";
      unlink(glob($cur_statsfile.".*"));
      push @{$batch_info->{"key"}}, $cur_keyfile;
    }
    $statsfile = $statsfile."_\$SGE_TASK_ID";
    $keyfile = $keyfile."_\$SGE_TASK_ID";
  } else {
    unlink(glob($statsfile.".*"));
    push @{$batch_info->{"key"}}, $keyfile;
    print $list_fh $statsfile."\n";
  }
  print $fh get_batch_script_pre_string($temp_dir, $temp_dir);
  print $fh "$BINDIR/stats -b $model_base -c $cfg -r $RECIPE -H -o $statsfile $STATS_MODE -F $FORWARD_BEAM -W $BACKWARD_BEAM -A $AC_SCALE $spkc_switch $batch_options -i $VERBOSITY\n";
  print $fh "touch $keyfile\n";
  close($fh);
  push @{$batch_info->{"script"}}, $scriptfile;
  close($list_fh);
  submit_and_wait($batch_info);
}


sub optimization_step {
  my $temp_dir = shift(@_);
  my $im = shift(@_);
  my $om = shift(@_);
  my $stats_list_file = shift(@_);
  my $init_options = shift(@_);

  my $batch_info = make_single_batch($temp_dir, $BASE_ID, "$BINDIR/optmodel -b $im -L $stats_list_file -o $om -F opt.state --bfgsu $NUM_BFGS_UPDATES --minvar $MINVAR -i $VERBOSITY -s ${BASE_ID}_summary $init_options\n");
  submit_and_wait($batch_info, 10); # Reduced batch check interval
}



###############################
# Aku-specific batch functions
###############################

sub get_aku_batch_options {
  my $num_batches = shift(@_);
  my $info = shift(@_);
  my $options = "";
  if ($num_batches > 1) {
    $info->{"qsub_options"} = "-t 1-$num_batches";
    $options = "-B $num_batches -I \$SGE_TASK_ID";
  }
  return $options;
}

###############################
# Generic batch functions
###############################

sub get_empty_batch_info {
  my $batch_info = {};
  $batch_info->{"script"} = [];
  $batch_info->{"key"} = [];
  $batch_info->{"qsub_options"} = "";
  return $batch_info;
}

sub get_batch_script_pre_string {
  my $script_dir = shift(@_);
  my $out_dir = shift(@_);
  return "#!/bin/sh\n#\$ -S /bin/sh\n#\$ -o ${out_dir}\n#\$ -e ${out_dir}\ncd ${script_dir}\n"
}

sub make_single_batch {
  my $temp_dir = shift(@_);
  my $script_id = shift(@_);
  my $script_cmd = shift(@_);
  my $batch_info = get_empty_batch_info();
  my ($scriptfile, $keyfile);
  my $fh;
  $scriptfile = "single_${script_id}.sh";
  $keyfile = "${temp_dir}/single_${script_id}_ready";
  open $fh, "> $scriptfile" || die "Could not open $scriptfile";
  print $fh get_batch_script_pre_string($temp_dir, $temp_dir);
  print $fh $script_cmd."\n";
  print $fh "touch $keyfile\n";
  close($fh);
  push @{$batch_info->{"script"}}, $scriptfile;
  push @{$batch_info->{"key"}}, $keyfile;
  return $batch_info;
}

sub submit_and_wait {
  my $batch_info = shift(@_);
  my $batch_check_interval = shift(@_); # In seconds
  $batch_check_interval = 100 if (!(defined $batch_check_interval));

  for my $i (0..scalar @{$batch_info->{"key"}}-1) {
    system("rm ".${$batch_info->{"key"}}[$i]) if (-e ${$batch_info->{"key"}}[$i]);
  }

  for my $i (0..scalar @{$batch_info->{"script"}}-1) {
    my $qsub_command = "qsub ".$batch_info->{"qsub_options"}." ".${$batch_info->{"script"}}[$i];
    system("chmod u+x ".${$batch_info->{"script"}}[$i]);
    system("$qsub_command") && die("Error in '$qsub_command'\n");
  }
  my @sub_ready = ();
  my $ready_count = 0;
  while ($ready_count < scalar @{$batch_info->{"key"}}) {
    sleep($batch_check_interval);
    for my $i (0..scalar @{$batch_info->{"key"}}-1) {
      if (!$sub_ready[$i]) {
        if (-e ${$batch_info->{"key"}}[$i]) {
          $sub_ready[$i] = 1;
          $ready_count++;
        }
      }
    }
  }
  for my $i (0..scalar @{$batch_info->{"key"}}-1) {
    system("rm ".${$batch_info->{"key"}}[$i]);
  }
}
