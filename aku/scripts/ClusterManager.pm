package ClusterManager;

use strict;

sub new {
    my ($type) = @_;
    my $self = {};

    $self->{"identifier"} = "CM";
    $self->{"priority"} = 0; # 0 = fastest, 10000=slowest
    $self->{"mem_req"} = 0; # In megabytes
    $self->{"run_time"} = 0; # In minutes
    $self->{"jobs"} = []; # Currently active jobs
    $self->{"first_batch"} = 0; # If with last_batch zero, single mode is used
    $self->{"last_batch"} = 0;
    $self->{"run_dir"} = ".";
    $self->{"log_dir"} = ".";
    $self->{"batch_check_interval"} = 20; # Seconds
    $self->{"shutdown_on_failure"} = 1;
    $self->{"failed_batch_retry_count"} = 0;
    $self->{"interrupted"} = 0;
    $self->{"exclude_nodes"} = "";

    bless $self, $type;
    return $self;
}


sub shutdown_jobs {
  my ($self) = @_;
  my $jobs = $self->{"jobs"};
  print "Shutting down ".($#{$jobs}+1)." jobs...\n";
  
  for (my $i = 0; $i <= $#{$jobs}; $i++) {
    system("scancel ".${$jobs}[$i]{job_id});
  }

  $self->{"jobs"} = [];
}


sub submit_batches {
  my ($self, $first_batch, $last_batch, $wrapper, $record_jobs) = @_;

  my $submit_command = "sbatch --no-requeue --partition=batch";
  $submit_command = $submit_command." --nice=".$self->{"priority"} if ($self->{"priority"} > 0);
  $submit_command = $submit_command." --mem-per-cpu=".$self->{"mem_req"} if ($self->{"mem_req"} > 0);
  $submit_command = $submit_command." --time=".$self->{"run_time"} if ($self->{"run_time"} > 0);
  $submit_command = $submit_command." -o ".$self->{"log_dir"}."/".$self->{"identifier"}.".stdout.\%B\%";
  $submit_command = $submit_command." -e ".$self->{"log_dir"}."/".$self->{"identifier"}.".stderr.\%B\%";
  if (length $self->{"exclude_nodes"}) {
    $submit_command = $submit_command . " --exclude='" . $self->{"exclude_nodes"} . "'";
  }
  $submit_command = $submit_command." $wrapper";

  print "Submit command: ".$submit_command."\n";

  for (my $i = $first_batch; $i <= $last_batch; $i++) {
    my $cur_submit = "$submit_command $i";
    if ($i == 0 && $last_batch == 0) {
      $cur_submit =~ s/\.\%B\%//g;
    } else {
      $cur_submit =~ s/\%B\%/$i/g;
    }
    my $new_job_number = -1;
    my $submit_counter = 0;
    while (1) {
      my $submit_result = `$cur_submit`;
      if ($submit_result =~ /Submitted batch job (\d+)/) {
        $new_job_number = $1;
        last;
      }
      if ($self->{"interrupted"}) {
        $self->shutdown_jobs();
        exit(1);
      }
      if (++$submit_counter >= 50) {
        print "Unable to submit job $i\n";
        $self->shutdown_jobs();
        exit(1);
      }
      my $wait_time = $submit_counter*2;
      print "Submission failed: exit code $?\n";
      print "Error message: $submit_result\n";
      print "Retrying in ${wait_time}s\n";
      sleep($wait_time);
    }
    if ($new_job_number < 0) {
      print "Invalid job number $new_job_number\n";
      $self->shutdown_jobs();
      exit(1);
    }
    # Write a grant file
    my $grant_file = $self->{"run_dir"}."/".$self->{"identifier"}.".grant.";
    my $fh;
    open $fh, "> ${grant_file}${i}" || die "Could not open ${grant_file}${i}";
    print $fh "$new_job_number\n";
    close($fh);

    if ($record_jobs) {
      push @{$self->{"jobs"}}, {job_id => $new_job_number, batch_id => $i, retry_count => 0};
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
  my $job_script_header = "#!/bin/sh\n";
  #my $job_script_header = "#!/bin/sh\n#\$ -S /bin/sh\n#\$ -o ".$self->{"log_dir"}."\n#\$ -e ".$self->{"log_dir"}."\n";
  my $script_wrapper = $self->{"run_dir"}."/".$self->{"identifier"}."_wrapper.sh";
  my $script_runner = $self->{"run_dir"}."/".$self->{"identifier"}."_runner.sh";
  my $epilog_script_name = $self->{"run_dir"}."/".$self->{"identifier"}.".epilog.sh";

  my $fh;
  my $single_mode = 0;
  my $readyfile;

  # Some sanity checks
  die "Invalid identifier: $self->{identifier}" if (!($self->{"identifier"} =~ /^\s*[\w\.]+\s*$/));

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

  # Write the script
  open $fh, "> $script_runner" || die "Could not open script file $script_runner";
  print $fh "#!/bin/sh\n";
  print $fh "BATCH=\$1\n";
  print $fh $script."\n";
  close($fh);
  system("chmod u+x ".$script_runner);

  open $fh, "> $script_wrapper" || die "Could not open $script_wrapper";
  print $fh $job_script_header;
  print $fh "BATCH=\$1\n";
  print $fh "cd ".$self->{"run_dir"}."\n";

  print $fh "echo SLURM_JOB_ID \$SLURM_JOB_ID\n";

  # Check that the job is valid, needed due to "Socket timed out" errors
  my $grant_file = $self->{"identifier"}.".grant.";
  print $fh "sleep 3s\n";
  print $fh "NUM=0; while [ ! -e ${grant_file}\$BATCH ] ; do sleep 5s; if [ \$NUM -gt 40 ]; then echo No grant file; exit; fi; NUM=\$[\$NUM + 1]; done\n";
  print $fh "jobid=`cat ${grant_file}\$BATCH`; if [ \$jobid != \$SLURM_JOB_ID ]; then echo Invalid job ID; exit; fi\n";
  print $fh "rm ${grant_file}\$BATCH\n";

  if (!$single_mode) {
    print $fh "echo \"Starting batch \$BATCH\"\n";
  }
  print $fh "echo \"Running on \`uname -n\`\"\n";
  print $fh "echo \"Job ID \$SLURM_JOB_ID\"\n";
  print $fh "export LD_LIBRARY_PATH=/triton/ics/project/puhe/support/lib:\$LD_LIBRARY_PATH\n";
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
  close($fh);
  system("chmod u+x ".$script_wrapper);

  open $fh, "> $epilog_script_name" || die "Could not open $epilog_script_name";
  print $fh "#!/bin/bash\n";
  print $fh "BATCH=\$1\n";
  print $fh "RUNNING_JOBS=\$2\n";
  print $fh "cd ".$self->{"run_dir"}."\n";
  print $fh $epilog_script."\n";
  close($fh);
  system("chmod u+x ".$epilog_script_name);

  # Set interrupt signal handler
  my $old_sig_int = $SIG{INT};
  $SIG{INT} = sub {$self->interrupt_handler};

  # Execute
  if ($single_mode) {
    # Remove possibly existing ready-file
    unlink($readyfile);
    # Submits just one batch
    $self->submit_batches(0, 0, $script_wrapper, $wait);
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
      unlink $self->{"identifier"}."_$i.ready";
    }
    
    $self->submit_batches($self->{"first_batch"}, $self->{"last_batch"},
                          $script_wrapper, $wait);

    if ($wait) {

      sleep($self->{"batch_check_interval"});

      while (!$self->{"interrupted"}) {

        # Just test with sacct --format=JobID,State,ExitCode -j ... ???
        # However, it doesn't show pending jobs.
        # What about other parallel systems?

        my $ready_file_mask = $self->{"run_dir"}."/".$self->{"identifier"}."_*.ready";
        my @ready_files = glob($ready_file_mask);
        if (!@ready_files) {
          sleep($self->{"batch_check_interval"});
          next;
        }
        
        my $jobs = $self->{"jobs"};
        for (my $i = 0; $i <= $#{$jobs}; $i++) {
          # Check if the batch has finished
          my $job_id = ${$jobs}[$i]{job_id};
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
              if ($self->{"failed_batch_retry_count"} > 0) {
                if ($retry_count >= $self->{"failed_batch_retry_count"}) {
                  print "Retry of batch $batch failed, skipping\n";
                } else {
                  print "Retrying, submitting batch $batch\n";
                  my $ji=$self->submit_batches($batch, $batch, $script_wrapper, 1);
                  ${$jobs}[$ji]{retry_count} = $retry_count + 1;
                  next;
                }
              }
              if ($self->{"shutdown_on_failure"}) {
                $SIG{INT} = $old_sig_int; # Restore the old signal handler
                $self->shutdown_jobs();
                exit(1);
              }
            } elsif (!/^OK$/) {
              print "Warning: Unknown end status for batch ".$batch." (job ".$job_id.")\n";
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
    kill('INT', $$); # Propagate
  }
}

1;
