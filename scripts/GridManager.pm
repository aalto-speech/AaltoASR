package GridManager;

use strict;

sub new {
    my ($type) = @_;
    my $self = {};

    $self->{"identifier"} = "GM";
    $self->{"qsub_options"} = "-q helli.q"; # Currently forced to helli.q
    $self->{"priority"} = 0;
    $self->{"mem_req"} = 0; # In megabytes
    $self->{"job_numbers"} = [];
    $self->{"running_batches"} = [];
    $self->{"retry_counters"} = [];
    $self->{"first_batch"} = 0; # If with last_batch zero, single mode is used
    $self->{"last_batch"} = 0;
    $self->{"run_dir"} = ".";
    $self->{"log_dir"} = ".";
    $self->{"batch_check_interval"} = 20; # Seconds
    $self->{"emergency_shutdown"} = 1;
    $self->{"failed_batch_retry_count"} = 0;

    bless $self, $type;
    return $self;
}


sub shutdown_jobs {
  my ($self) = @_;
  my $jobs = $self->{"job_numbers"};
  for (my $i = 0; $i <= $#{$jobs}; $i++) {
    print "Closing job ".${$jobs}[$i]."\n";
    system("qdel ${$jobs}[$i]");
  }

  $self->{"job_numbers"} = [];
  $self->{"running_batches"} = [];
  $self->{"retry_counters"} = [];
}


sub submit_batches {
  my ($self, $first_batch, $last_batch, $wrapper) = @_;
  my $qsub_command = "qsub";
  $qsub_command = $qsub_command." -p ".$self->{"priority"} if ($self->{"priority"} < 0);
  $qsub_command = $qsub_command." -l mem=".$self->{"mem_req"}."M" if ($self->{"mem_req"} > 0);
  $qsub_command = $qsub_command." -t ${first_batch}-${last_batch} ".$self->{"qsub_options"}." $wrapper";
  my $qsub_result = `$qsub_command`;
  print $qsub_result;
  if (!($qsub_result =~ /Your job(\-array)? (\d+)[\. ]/)) {
    print "Unable to submit the jobs\n";
    exit(1);
  }
  return $2;
}


sub submit {
  my ($self, $script) = @_;
  my $job_script_header = "#!/bin/sh\n#\$ -S /bin/sh\n#\$ -o ".$self->{"log_dir"}."\n#\$ -e ".$self->{"log_dir"}."\n";
  my $script_wrapper = $self->{"run_dir"}."/".$self->{"identifier"}."_wrapper.sh";
  my $script_runner = $self->{"run_dir"}."/".$self->{"identifier"}."_runner.sh";

  my $fh;
  my $single_mode = 0;
  my $readyfile;

  # Some sanity checks
  die "Invalid identifier" if (!($self->{"identifier"} =~ /^\s*\w+\s*$/));

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
    $readyfile = $self->{"identifier"}."_\$SGE_TASK_ID.ready";
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
  print $fh "cd ".$self->{"run_dir"}."\n";
  if (!$single_mode) {
    print $fh "echo \"Starting batch \$SGE_TASK_ID\"\n";
  }
  print $fh "echo \"Running on \`uname -n\`\"\n";
  print $fh "export LD_LIBRARY_PATH=/share/puhe/x86_64/lib/\n";
  print $fh "if $script_runner";
  print $fh " \$SGE_TASK_ID" if (!$single_mode);
  print $fh "; then echo \"OK\" > $readyfile; else echo \"FAILED\" > $readyfile; fi\n";
  close($fh);
  system("chmod u+x ".$script_wrapper);

  # Execute
  if ($single_mode) {
    # Remove possibly existing ready-file
    unlink($readyfile);
    my $qsub_command = "qsub ";
    $qsub_command = $qsub_command." -p ".$self->{"priority"} if ($self->{"priority"} < 0);
    $qsub_command = $qsub_command." -l mem=".$self->{"mem_req"}."M" if ($self->{"mem_req"} > 0);

    system($qsub_command." ".$self->{"qsub_options"}." $script_wrapper") && die("Error in qsub");
    while (1) {
      sleep($self->{"batch_check_interval"});
      if (-e $readyfile) {
        unlink($readyfile);
        last;
      }
    }
  } else {
    print "Starting batches ".$self->{"first_batch"}."-".
        $self->{"last_batch"}."\n";

    # Delete old ready files and initialize the tables
    for (my $i = $self->{"first_batch"}; $i <= $self->{"last_batch"}; $i++) {
      unlink $self->{"identifier"}."_$i.ready";
      push(@{$self->{"running_batches"}}, $i);
      ${$self->{"retry_counters"}}[$i] = 0;
    }
    
    my $job_num = $self->submit_batches($self->{"first_batch"}, 
                                        $self->{"last_batch"},
                                        $script_wrapper);
    push(@{$self->{"job_numbers"}}, $job_num);

    sleep($self->{"batch_check_interval"});

    while (1) {
      my $running_batches = $self->{"running_batches"};
      for (my $i = 0; $i <= $#{$running_batches}; $i++) {
        # Check if the batch has finished
        my $batch = ${$running_batches}[$i];
        my $ready_file = $self->{"run_dir"}."/".$self->{"identifier"}."_$batch.ready";
        if (-e $ready_file) {
          # Error checking
          my $ready_fh;
          open $ready_fh, "< $ready_file" || die "Could not open file $ready_file";
          $_=<$ready_fh>;
          close $ready_fh;
          unlink $ready_file;
          if (!/^OK$/) {
            print "Error in batch $batch\n";
            if ($self->{"failed_batch_retry_count"} > 0) {
              if ($self->{"retry_counters"}->[$batch] >= $self->{"failed_batch_retry_count"}) {
                print "Retry of batch $batch failed, skipping\n";
              } else {
                print "Retrying, submitting batch $batch\n";
                $self->{"retry_counters"}->[$batch]++;
                my $jnum = $self->submit_batches($batch, $batch, $script_wrapper);
                push(@{$self->{"job_numbers"}}, $jnum);
                next;
              }
            }
            if ($self->{"emergency_shutdown"}) {
              print "Shutting down\n";
              $self->shutdown_jobs();
              exit(1);
            }
          }

          @{$running_batches} =
              @{$running_batches}[0..($i-1),($i+1)..$#{$running_batches}];
          $i--;
        }
      }

      last if (!@{$running_batches}); # Done
      
      sleep($self->{"batch_check_interval"});
    }
  }
}

1;
