#!/usr/bin/perl

# Run this script at itl-cl1, as it uses GridEngine for scheduling
# the parallel processes.

use locale;
use strict;

# Model name
my $ID="mfcc_mmi";

# Path settings
my $AKUBINDIR="/home/".$ENV{"USER"}."/aku";
my $SCRIPTDIR="/home/".$ENV{"USER"}."/aku/scripts";
my $HMMDIR="/share/puhe/".$ENV{"USER"}."/hmms";
my $workdir="/share/work/".$ENV{"USER"}."/aku_work";

# Training file list
my $RECIPE="/share/puhe/jpylkkon/speecon_new/speecon_new_train.recipe";

my $LATTICERECSCRIPT="/share/puhe/jpylkkon/speecon_new/rec_lattice.py";

my $HMMMODEL="$HMMDIR/speecon_new_final_ml";
my $LEXICON="/share/puhe/jpylkkon/speecon_new/morph19k.lex";
my $LMMODEL="/share/work/jpylkkon/bin_lm/morph19k_2gram.bin";
my $VOCABULARY;

my $USE_MORPHS = 1; # 1 for morph language models, 0 for word based LMs

# Batch settings
my $NUM_BATCHES = 4; # Number of processes in parallel
my $NUM_BLOCKS = 50; # In how many blocks the data is generated in one batch

my $LATTICE_THRESHOLD = 0.00001; # Lattice pruning threshold, [0, 1)
my $LMSCALE = 32;

# $LNA_OPTIONS must include -R in case of raw audio files!
my $LNA_OPTIONS = "\'-R --clusters ${HMMMODEL}.gcl --eval-ming=0.1\'";

my $tempdir = $workdir."/".$ID;
mkdir $tempdir;
chdir $tempdir || die("Could not chdir to $tempdir");

$VOCABULARY=$tempdir."/vocabulary.tmp";
binlm_to_vocabulary($LMMODEL, $VOCABULARY);

my $batch_info = make_single_batch($tempdir, $ID,
"export PERL5LIB=$SCRIPTDIR\n".
"$SCRIPTDIR/lex2fst.pl $LEXICON > L.fst\n".
"$SCRIPTDIR/hmms2fsm.pl ${HMMMODEL}.ph . | fst_closure -p - H.fst\n" .
"$SCRIPTDIR/hmms2trinet.pl ${HMMMODEL}.ph . | fst_optimize -A - - > C.fst\n");
submit_and_wait($batch_info, 10);

my $fh;

open $fh, "> sentence_end.fst";
print $fh "#FSTBasic MaxPlus\nI 0\nF 1\nT 0 1 </s> </s>\n";
close $fh;

open $fh, "> optional_silence.fst";
print $fh "#FSTBasic MaxPlus\nI 0\nF 1\nT 0 1 __ __\nT 0 1\n";
close $fh;

my $scriptfile = $ID."_gen_den.sh";
open $fh, "> $scriptfile" || die "Could not open $scriptfile";
print $fh get_batch_script_pre_string($tempdir, $tempdir);
print $fh "$SCRIPTDIR/den_hmmnet_batch_sub.pl \$SGE_TASK_ID $NUM_BATCHES $NUM_BLOCKS $USE_MORPHS $HMMMODEL $LEXICON $LMMODEL $VOCABULARY $RECIPE $AKUBINDIR $SCRIPTDIR $LNA_OPTIONS $LATTICERECSCRIPT $LATTICE_THRESHOLD $LMSCALE\n";
close($fh);
system("qsub -t 1-$NUM_BATCHES $scriptfile") == 0 || die "qsub failed\n";


sub binlm_to_vocabulary {
  my $lm = shift(@_);
  my $vocabfile = shift(@_);
  my $in_fh;
  my $out_fh;
  my $num_tokens;

  open $in_fh, "< $lm";
  open $out_fh, "> $vocabfile";

  $_=<$in_fh>;
  $_=<$in_fh>;
  $num_tokens = <$in_fh>;
  for (my $i = 0; $i < $num_tokens; $i++) {
    $_=<$in_fh>;
    print $out_fh $_;
  }
  close($out_fh);
  close($in_fh);
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
