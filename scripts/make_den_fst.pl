#!/usr/bin/perl

# DO NOT RUN DIRECTLY, USE generate_den_hmmnets.pl INSTEAD!

use locale;
use strict;
use BSD::Resource;

$SIG{PIPE} = 'IGNORE';
setrlimit(RLIMIT_CORE, 0, 0); # Repress core dumps


my $USE_MORPHS = shift @ARGV;
my $VOCABULARY = shift @ARGV;
my $LMMODEL = shift @ARGV;
my $RECIPE = shift @ARGV;
my $TEMPDIR = shift @ARGV;
my $SCRIPTDIR = shift @ARGV;
my $NUM_BATCHES = shift @ARGV;
my $BATCH_INDEX = shift @ARGV;
my $LATTICE_THRESHOLD = shift @ARGV;
my $LMSCALE = shift @ARGV;

my @BATCH_FILES;


# Binaries
my $LATTICE_RESCORE = "/home/jpylkkon/aku/lattice_rescore/lattice_rescore";
my $SRI_LATTICE_TOOL = "/share/puhe/srilm-1.5.1/bin/i686-m64/lattice-tool";
my $MORPH_LATTICE = "/home/thirsima/Work/morph_lattice/morph_lattice";

# Scripts
my $FSM2HTK = "$SCRIPTDIR/fsm2htk.pl";
my $PHN2TRANSCRIPT = "$SCRIPTDIR/phn2transcript.pl";
my $HTK2FST = "$SCRIPTDIR/htk2fst.pl";
my $RMB = "$SCRIPTDIR/rmb.pl";
my $TRANSCRIPT2FSM = "$SCRIPTDIR/transcript2fsm.pl";


load_recipe($RECIPE, $BATCH_INDEX, $NUM_BATCHES);
make_transcript_word_fsts();
make_denominator_hmmnets();


sub load_recipe {
  my $recipe_file = shift(@_);
  my $batch_index = shift(@_);
  my $num_batches = shift(@_);
  my $target_lines;
  my $fh;
  my @recipe_lines;
  
  open $fh, "< $recipe_file" || die "Could not open $recipe_file\n";
  @recipe_lines = <$fh>;
  close $fh;

  if ($num_batches <= 1) {
    $target_lines = $#recipe_lines;
  } else {
    $target_lines = int($#recipe_lines/$num_batches);
    $target_lines = 1 if ($target_lines < 1);
  }
  
  my $cur_index = 1;
  my $cur_line = 0;

  foreach my $line (@recipe_lines) {
    if ($num_batches > 1 && $cur_index < $num_batches) {
      if ($cur_line >= $target_lines) {
        $cur_index++;
        last if ($cur_index > $batch_index);
        $cur_line -= $target_lines;
      }
    }

    if ($num_batches <= 1 || $cur_index == $batch_index) {
      $line =~ /transcript=(\S*)/;
      my $trfile = $1;
      $trfile =~ /.*\/([^\/]+)$/;
      # Augment with line number to prevent overwriting in case of same
      # file names
      my $trhmmnet = $1."_${cur_line}.tmptr";
      $line =~ /lna=(\S*)/;
      my $wgfile = $TEMPDIR."/".$1.".wg";
      $line =~ /den\-hmmnet=(\S*)/;
      my $denfile = $1;

      push(@BATCH_FILES, [$trfile, $trhmmnet, $wgfile, $denfile]);
    }
    $cur_line++;
  }
}


sub make_transcript_word_fsts {
  my $list_file = "filelist.tmp";
  my $l;

  chdir($TEMPDIR) || die "Could not change to directory $TEMPDIR";

  open(LIST, "> $list_file");
  for $l (@BATCH_FILES) {
    print LIST $l->[1]."\n";
    if ($USE_MORPHS) {
      system("$PHN2TRANSCRIPT ".$l->[0]." | $MORPH_LATTICE $VOCABULARY - - | $FSM2HTK > ".$l->[1]) == 0 || die "system error $7\n";
    } else {
      print "$PHN2TRANSCRIPT ".$l->[0]." | $TRANSCRIPT2FSM | $FSM2HTK > ".$l->[1]."\n";
      system("$PHN2TRANSCRIPT ".$l->[0]." | $TRANSCRIPT2FSM | $FSM2HTK > ".$l->[1]) == 0 || die "system error $7\n";
    }
  }
  close(LIST);

  my $temp_out_dir = "out";
  die "Temporary output directory already exists!" if (-e $temp_out_dir);
  mkdir($temp_out_dir);

  system("$LATTICE_RESCORE -l $LMMODEL -I $list_file -O $temp_out_dir");

  for my $l (@BATCH_FILES) {
    print "Processing ".$l->[1]."\n";
    my $c;
    for ($c = 0; $c < 4; $c++) {
      my $r;
      $r = system("$HTK2FST ${temp_out_dir}/".$l->[1]." | fst_nbest -t -1000 -n 1 -p - ".$l->[1]);
      last if ($r == 0);
    }
    die "Could not process file ".$l->[1]."\n" if ($c >= 4);
  }

  system("rm -rf $temp_out_dir\n"); # Remove temporary files

  chdir("..");
}


sub make_denominator_hmmnets {
  my $l;
  my $c;
  my $r;

  for $l (@BATCH_FILES) {
    print "Generating denominator hmmnet for file ".$l->[2]."\n";
    for ($c = 0; $c < 4; $c++) {
      $r = system("$LATTICE_RESCORE -l $LMMODEL -i ".$l->[2]." -o - | $SRI_LATTICE_TOOL -posterior-prune $LATTICE_THRESHOLD -read-htk -in-lattice - -write-htk -out-lattice - -htk-lmscale $LMSCALE -htk-acscale 1 -posterior-scale $LMSCALE | $HTK2FST - | fst_concatenate optional_silence.fst - - | fst_concatenate - optional_silence.fst - | fst_concatenate - sentence_end.fst - | fst_union ${TEMPDIR}/".$l->[1]." - - | fst_optimize -A - - | fst_compose -t L.fst - - | fst_optimize -A - - | fst_compose -t C.fst - - | fst_optimize -A - - | fst_compose -t H.fst - - | $RMB | fst_optimize -A - ".$l->[3]);
      last if ($r == 0);
    }
    die "Could not process file ".$l->[2]."\n" if ($c >= 10);
  }
}
