package CondorManager;

use strict;

sub new {
    my ($type) = @_;
    my $self = {};

    $self->{"identifier"} = "CM";
    $self->{"priority"} = 0; # 0 = fastest, 10000=slowest. FIXME: NOT USED!
    $self->{"mem_req"} = 0; # In megabytes. FIXME: NOT USED!
    $self->{"run_time"} = 0; # In minutes. FIXME: NOT USED!
    $self->{"jobs"} = []; # Currently active jobs
    $self->{"first_batch"} = 0; # If with last_batch zero, single mode is used
    $self->{"last_batch"} = 0;
    $self->{"run_dir"} = ".";
    $self->{"log_dir"} = ".";
    $self->{"batch_check_interval"} = 20; # Seconds
    $self->{"shutdown_on_failure"} = 1;
    $self->{"failed_batch_retry_count"} = 0; #FIXME: NOT USED!
    $self->{"interrupted"} = 0;
    $self->{"cur_condor_cluster"} = -1;

    bless $self, $type;
    return $self;
}


sub shutdown_jobs {
  my ($self) = @_;
  my $jobs = $self->{"jobs"};

  if ($self->{"cur_condor_cluster"} >= 0) {
    print "Shutting down ".($#{$jobs}+1)." jobs...\n";
    system("condor_rm ".$self->{"cur_condor_cluster"});
  }

  $self->{"cur_condor_cluster"} = -1;
  $self->{"jobs"} = [];
}


sub submit_batches {
  my ($self, $first_batch, $last_batch, $script, $record_jobs) = @_;

  my $submit_command = "condor_submit";
  $submit_command = $submit_command." $script";

  print "Submit command: ".$submit_command."\n";
  my $submit_result = `$submit_command`;
  if (!($submit_result =~ /submitted to cluster (\d+)\./)) {
    print "Unable to submit the jobs:\n";
    print $submit_result;
    exit(1);
  }
  $self->{"cur_condor_cluster"} = $1;

  for (my $i = $first_batch; $i <= $last_batch; $i++) {
    if ($record_jobs) {
      push @{$self->{"jobs"}}, {batch_id => $i, retry_count => 0};
    }
  }

  print "".($last_batch - $first_batch+1);
  if ($last_batch > $first_batch) {
    print " batches submitted successfully\n";
  } else {
    print " batch submitted successfully\n";
  }

  return $#{$self->{"jobs"}};
}


sub interrupt_handler {
    my ($self) = @_;
    $self->{"interrupted"} = 1;
}


sub submit {
  my ($self, $script, $epilog_script, $wait) = @_;
  $wait = 1 if (!defined $wait);
  my $script_wrapper = $self->{"run_dir"}."/".$self->{"identifier"}."_wrapper.sh";
  my $script_runner = $self->{"run_dir"}."/".$self->{"identifier"}."_runner.sh";
  my $submit_script = $self->{"run_dir"}."/".$self->{"identifier"}.".submit";
  my $epilog_script_name = $self->{"run_dir"}."/".$self->{"identifier"}.".epilog.sh";

  my $fh;
  my $single_mode = 0;
  my $readyfile;

  # Some sanity checks
  die "Invalid identifier" if (!($self->{"identifier"} =~ /^\s*[\w\.]+\s*$/));

  if (($self->{"first_batch"} != 0 || $self->{"last_batch"} != 0) &&
      $self->{"first_batch"} < 1 ||
      $self->{"first_batch"} > $self->{"last_batch"}) {
    die "Invalid batch numbers: first = ".$self->{"first_batch"}.", last = ".$self->{"last_batch"}."\n";
  }

  # Check for single mode
  if ($self->{"first_batch"} == 0 && $self->{"last_batch"} == 0) {
    $single_mode = 1;
    $readyfile = $self->{"identifier"}.".ready";
  } else {
    $readyfile = $self->{"identifier"}."_\$BATCH.ready";
  }

  # Write the scripts
  open $fh, "> $script_runner" || die "Could not open script file $script_runner";
  print $fh "#!/bin/bash\n";
  print $fh "BATCH=\$1\n";
  print $fh $script."\n";
  close($fh);
  system("chmod u+x ".$script_runner);

  open $fh, "> $script_wrapper" || die "Could not open $script_wrapper";
  print $fh "#!/bin/bash --login\n"; # Ensure the reading of configuration files
  print $fh "BATCH=\$1\n";
  print $fh "BATCH=\$(( BATCH + ".$self->{"first_batch"}."))\n";
  print $fh "cd ".$self->{"run_dir"}."\n";
  print $fh "echo \"Running on `uname -n`\"\n";

  if ($wait) {
    print $fh "if $script_runner";
    print $fh " \$BATCH" if (!$single_mode);
    print $fh "; then echo \"OK\" > $readyfile; else echo \"FAILED\" > $readyfile; fi\n";
  } else {
    if ($single_mode) {
      print $fh "$script_runner\n";
    } else {
      print $fh "$script_runner \$BATCH\n";
    }
  }
  close $fh;
  system("chmod u+x ".$script_wrapper);

  open $fh, "> $epilog_script_name" || die "Could not open $epilog_script_name";
  print $fh "#!/bin/bash\n";
  print $fh "BATCH=\$1\n";
  print $fh "RUNNING_JOBS=\$2\n";
  print $fh "cd ".$self->{"run_dir"}."\n";
  print $fh $epilog_script."\n";
  close($fh);
  system("chmod u+x ".$epilog_script_name);

  open $fh, "> $submit_script" || die "Could not open $submit_script";
  print $fh "requirements = ICSLinux == 2010\n";
  print $fh "notification = Never\n";
  print $fh "environment = \"LD_LIBRARY_PATH=/home/jpylkkon/local/lib\"\n";
  print $fh "arguments = \$(Process)\n";
  print $fh "executable = $script_wrapper\n";
  print $fh "log = ".$self->{"log_dir"}."/".$self->{"identifier"}.".condor.\$(Cluster)\n";
  print $fh "output = ".$self->{"log_dir"}."/".$self->{"identifier"}.".stdout.\$(Process)\n";
  print $fh "error = ".$self->{"log_dir"}."/".$self->{"identifier"}.".stderr.\$(Process)\n";
  print $fh "rank = KeyboardIdle\n";

  if (!$single_mode) {
    print $fh "queue ".($self->{"last_batch"}-$self->{"first_batch"}+1)."\n";
  } else {
    print $fh "queue";
  }

  close($fh);

  # Set interrupt signal handler
  my $old_sig_int = $SIG{INT};
  $SIG{INT} = sub {$self->interrupt_handler};

  # Execute
  if ($single_mode) {
    # Remove possibly existing ready-file
    unlink($readyfile);
    # Submits just one batch
    $self->submit_batches(0, 0, $submit_script, $wait);
    if ($wait) {
      while (!$self->{"interrupted"}) {
        sleep($self->{"batch_check_interval"});
        if (-e $readyfile) {
          unlink($readyfile);
          $self->{"jobs"} = [];
          last;
        }
      }
    }
  } else {
    print "Starting batches ".$self->{"first_batch"}."-".
        $self->{"last_batch"}."\n";

    # Delete old ready files
    for (my $i = $self->{"first_batch"}; $i <= $self->{"last_batch"}; $i++) {
      unlink $self->{"run_dir"}."/".$self->{"identifier"}."_$i.ready";
    }
    
    $self->submit_batches($self->{"first_batch"}, $self->{"last_batch"},
                          $submit_script, $wait);

    if ($wait) {

      sleep($self->{"batch_check_interval"});

      while (!$self->{"interrupted"}) {

        my $ready_file_mask = $self->{"run_dir"}."/".$self->{"identifier"}."_*.ready";
        my @ready_files = glob($ready_file_mask);
        if (!@ready_files) {
          sleep($self->{"batch_check_interval"});
          next;
        }
        
        my $jobs = $self->{"jobs"};
        for (my $i = 0; $i <= $#{$jobs}; $i++) {
          # Check if the batch has finished
          my $batch = ${$jobs}[$i]{batch_id};
          my $retry_count = ${$jobs}[$i]{retry_count};
          my $ready_file = $self->{"run_dir"}."/".$self->{"identifier"}."_$batch.ready";
          if (-e $ready_file) {
            # Remove the job from the active job array
            @{$jobs} = @{$jobs}[0..($i-1),($i+1)..$#{$jobs}];
            $i--;

            # Error checking
            for (my $k=0; $k < 10; $k++) {
              my $ready_fh;
              open $ready_fh, "< $ready_file" || die "Could not open file $ready_file";
              $_=<$ready_fh>;
              close $ready_fh;
              last if (length($_) > 0);
              sleep(10);
            }
            unlink $ready_file;
            if (/^FAILED$/) {
              print "Error in batch $batch\n";
              print "\"".$_."\"\n";
              # if ($self->{"failed_batch_retry_count"} > 0) {
              #   if ($retry_count >= $self->{"failed_batch_retry_count"}) {
              #     print "Retry of batch $batch failed, skipping\n";
              #   } else {
              #     print "Retrying, submitting batch $batch\n";
              #     my $ji=$self->submit_batches($batch, $batch, $script_wrapper, 1);
              #     ${$jobs}[$ji]{retry_count} = $retry_count + 1;
              #     next;
              #   }
              # }
              if ($self->{"shutdown_on_failure"}) {
                $SIG{INT} = $old_sig_int; # Restore the old signal handler
                $self->shutdown_jobs();
                exit(1);
              }
            } elsif (!/^OK$/) {
              print "Warning: Unknown end status for batch ".$batch."\n";
            }

            # Execute the epilog script
            my $jobs_running = @{$jobs};
            system("$epilog_script_name $batch $jobs_running") && die "Error in epilog script";
          }
        }

        last if (!@{$jobs}); # Done
      }
    }
  }
  $SIG{INT} = $old_sig_int; # Restore the old signal handler
  if ($self->{"interrupted"}) {
    $self->shutdown_jobs();
    #kill('INT', $$); # Propagate
  }
}

1;
