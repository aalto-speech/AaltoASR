#!/bin/bash
# Usage: build_helper_fsts.sh [-m] [-s SCRIPT_DIR] LEX-FILE PH-FILE

mswitch=""
if [[ $1 == "-m" ]]; then
    mswitch="-m"
    shift
fi

script_dir=""
if [[ $1 == "-s" ]]; then
    script_dir=${2}/
    PERL5LIB=$script_dir
    export PERL5LIB
    shift 2
fi

lex=$1
ph=$2

if [[ ${#ph} -eq 0 ]]; then
  echo Usage: build_helper_fsts.sh [-m] [-s SCRIPT_DIR] LEX-FILE PH-FILE
  exit
fi

# Build a vocabulary out of the lexicon
cut -f 1 -d '(' $lex | sort | uniq > ${lex}.voc

# Build the Helper FSTs
${script_dir}lex2fst.pl $mswitch $lex > L.fst
${script_dir}hmms2trinet.pl $ph . | fst_optimize -A - - > C.fst
${script_dir}hmms2fsm.pl $ph . | fst_closure -p - H.fst

echo "#FSTBasic MaxPlus" > optional_silence.fst
echo "I 0" >> optional_silence.fst
echo "F 3" >> optional_silence.fst
if [[ $mswitch == "-m" ]]; then
  echo "T 0 1 #1 <w>" >> optional_silence.fst
else
  echo "T 0 1 #1" >> optional_silence.fst
fi
echo "T 1 2 __" >> optional_silence.fst
echo "T 2 3 #1" >> optional_silence.fst
echo "T 0 3" >> optional_silence.fst

echo "#FSTBasic MaxPlus" > end_mark.fst
echo "I 0" >> end_mark.fst
echo "T 0 1 #1 #E" >> end_mark.fst
echo "F 1" >> end_mark.fst
