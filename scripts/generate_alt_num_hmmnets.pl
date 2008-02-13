#!/usr/bin/perl

# Assumes that the lexicon contains all single letters!

use locale;
use strict;

# Path settings
my $AKUBINDIR="/home/".$ENV{"USER"}."/aku";
my $SCRIPTDIR="/home/".$ENV{"USER"}."/aku/scripts";
my $HMMDIR="/share/puhe/".$ENV{"USER"}."/hmms";

# Training file list
my $RECIPE="/share/puhe/jpylkkon/speecon_new/speecon_new_train.recipe";

my $HMMMODEL="$HMMDIR/speecon_new_final_ml";
my $LEXICON="/share/puhe/jpylkkon/speecon_new/morph19k.lex";


$ENV{"PERL5LIB"}=$SCRIPTDIR;
system("$SCRIPTDIR/lex2fst.pl $LEXICON > L.fst") && die("Could not create L.fst");
system("$SCRIPTDIR/hmms2fsm.pl ${HMMMODEL}.ph . | fst_closure -p - H.fst") && die("Could not create H.fst");
system("$SCRIPTDIR/hmms2trinet.pl ${HMMMODEL}.ph . | fst_optimize -A - - > C.fst") && die("Could not create C.fst");


my $fh;

open $fh, "> sentence_end.fst";
print $fh "#FSTBasic MaxPlus\nI 0\nF 1\nT 0 1 </s> </s>\n";
close $fh;

open $fh, "> optional_silence.fst";
print $fh "#FSTBasic MaxPlus\nI 0\nF 1\nT 0 1 __ <w>\nT 0 1\n";
close $fh;

open $fh, "> end_symbol.fst";
print $fh "#FSTBasic MaxPlus\nI 0\nF 1\nT 0 1 ## ,\n";
close $fh;


open $fh, "< $RECIPE" || die "Could not open $RECIPE\n";

my $hmmnet;
my $phn;
my $c;

while (<$fh>) {
  /hmmnet=(\S*)/;
  $hmmnet = $1;
  /transcript=(\S*)/;
  $phn = $1;
  
  print "Generating hmmnet file ".$hmmnet."\n";
  for ($c = 0; $c < 8; $c++) {
    my $r;
    $r = system("$SCRIPTDIR/phn2transcript.pl $phn | $SCRIPTDIR/transcript2char_fst.pl | fst_concatenate - sentence_end.fst - | fst_optimize -A - - | fst_compose -t L.fst - - | fst_concatenate optional_silence.fst - - | fst_concatenate - optional_silence.fst - | fst_optimize -A - - | fst_compose -t C.fst - - | fst_project  i e - - | fst_optimize -A - - | fst_compose -t H.fst - - | $SCRIPTDIR/rmb.pl | fst_optimize -A - $hmmnet ");
    last if ($r == 0);
  }
  die "Could not generate file ".$hmmnet."\n" if ($c >= 8);

}

