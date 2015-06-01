#!/bin/bash -e
#
# Good for recognizing a couple of files specified in the command line. Writes
# LNAs to
#   $WORK_DIR/recognitions/<AM options>,
# and possibly lattices to
#   $WORK_DIR/recognitions/<AM options>/<decoder options>,
# Writes the recognitions results (hypotheses) under
#   $WORK_DIR/results
# and displays them on the screen.

SCRIPT_DIR=$(dirname "$0")

[ -n "$AM" ] || { echo "AM environment variable needs to be specified." >&2; exit 2; }
[ -n "$LM" ] || { echo "LM environment variable needs to be specified." >&2; exit 2; }
[ -n "$DICTIONARY" ] || { echo "DICTIONARY environment variable needs to be specified." >&2; exit 2; }
[ -n "$BEAM" ] || BEAM=280
[ -n "$LM_SCALE" ] || LM_SCALE=30
[ -n "$TOKEN_LIMIT" ] || TOKEN_LIMIT=100000
[ -n "$RECOGNITIONS_DIR" ] || RECOGNITIONS_DIR="$WORK_DIR/recognitions"
[ -n "$RESULTS_DIR" ] || RESULTS_DIR="$WORK_DIR/results"

AM_OPT=$(basename $AM)

DECODER_OPT=""
[ -n "$DECODER_VER" ] && DECODER_OPT="$DECODER_VER--"
DECODER_OPT="$DECODER_OPT$(basename $LM)"

PARAMS=""

[ -n "$AKU" ] && PARAMS="$PARAMS --aku $AKU"
[ -n "$DECODER" ] && PARAMS="$PARAMS --decoder $DECODER"

# Specify binary LM if .bin or .fsabin file exists.
if [ "$FSA" != "" ]
then
	PARAMS="$PARAMS --fsa"
	BIN_LM_PATH="$LM.fsabin"
	DECODER_OPT=$DECODER_OPT-fsa
else
	BIN_LM_PATH="$LM.bin"
fi
if [ -e "$BIN_LM_PATH" ]
then
	PARAMS="$PARAMS --bin-lm $BIN_LM_PATH"
elif [ ! -e "$LM" ]
then
	echo "Neither binary LM ($BIN_LM_PATH) nor ARPA LM ($LM) was found." >&2
	exit 2
fi

# ARPA LM may be needed in any case for rescoring lattices.
if [ -e "$LM" ]
then
	PARAMS="$PARAMS --arpa-lm $LM"
fi

# Specify binary lookahead LM if exists, otherwise ARPA.
if [ "$LOOKAHEAD_LM" != "" ]
then
	if [ -e "$LOOKAHEAD_LM.bin" ]
	then
		PARAMS="$PARAMS --lookahead-bin-lm $LOOKAHEAD_LM.bin"
	elif [ -e "$LOOKAHEAD_LM" ]
	then
		PARAMS="$PARAMS --lookahead-arpa-lm $LOOKAHEAD_LM"
	else
		echo "Lookahead LM does not exist: $LOOKAHEAD_LM" >&2
		exit 2
	fi
	
	# Remove the common prefix from lookahead LM name to shorten the file names.
	LOOKAHEAD_LM_UNIQUE=$(printf "%s\n%s\n" $(basename $LM) $(basename $LOOKAHEAD_LM) | sed -e 'N;s/^\(.*\).*\n\1//')
	DECODER_OPT="$DECODER_OPT--$(basename $LOOKAHEAD_LM_UNIQUE)"
fi

DECODER_OPT="$DECODER_OPT--$(basename $DICTIONARY)-"

if [ "$CLASSES" != "" ]
then
        PARAMS="$PARAMS --classes $CLASSES"
        DECODER_OPT="$DECODER_OPT-$(basename $CLASSES .sricls)-"
fi

if [ "$SPLIT_MULTIWORDS" != "" ]
then
	PARAMS="$PARAMS --split-multiwords"
	DECODER_OPT=$DECODER_OPT-mw
fi

if [ "$GENERATE_LATTICES" != "" ]
then
	PARAMS="$PARAMS --generate-word-graph"
	if [ "$LATTICE_TOOL" = "" ]
	then
		LATTICE_TOOL=$(which lattice-tool)
	fi
	if [ "$LATTICE_TOOL" != "" ]
	then
		PARAMS="$PARAMS --lattice-tool $LATTICE_TOOL"
	fi
	if [ "$LATTICE_RESCORE" != "" ]
	then
		PARAMS="$PARAMS --lattice-rescore $LATTICE_RESCORE"
	fi
	if [ "$WORD_PAIR_APPROXIMATION" != "" ]
	then
		PARAMS="$PARAMS --word-pair-approximation"
		DECODER_OPT=$DECODER_OPT-wpa
	fi
fi

DECODER_OPT=$DECODER_OPT-b$BEAM-s$LM_SCALE
DECODER_OPT=$DECODER_OPT-tl$(echo $TOKEN_LIMIT|"$SCRIPT_DIR/human-readable.awk")

if [ "$ADAPTATION" != "" ]
then
	PARAMS="$PARAMS --adapt $ADAPTATION"
	DECODER_OPT="$DECODER_OPT-$ADAPTATION"
	if [ "$SPEAKER_ID_FIELD" = "" ]
	then
		echo "Warning: speaker ID field not specified. Using default (3)." 2>&1
		PARAMS="$PARAMS --speaker-id-field 3"
	else
		PARAMS="$PARAMS --speaker-id-field $SPEAKER_ID_FIELD"
	fi
fi

WORK_DIR="$RECOGNITIONS_DIR/$AM_OPT"
REC_DIR="$WORK_DIR/$DECODER_OPT"
HYP_FILE="$RESULTS_DIR/recognize-files.trn"

mkdir -p "$REC_DIR"
mkdir -p "$RESULTS_DIR"
rm -f "$HYP_FILE"

PARAMS="$PARAMS --am $AM"
PARAMS="$PARAMS --dictionary $DICTIONARY.lex"
PARAMS="$PARAMS --beam $BEAM"
PARAMS="$PARAMS --language-model-scale $LM_SCALE"
PARAMS="$PARAMS --token-limit $TOKEN_LIMIT"
PARAMS="$PARAMS --hypothesis-file $HYP_FILE"
PARAMS="$PARAMS --work-directory $WORK_DIR"
PARAMS="$PARAMS --rec-directory $REC_DIR"
PARAMS="$PARAMS --verbose 1"
PARAMS="$PARAMS $*"

# Uncomment for debugging:
#gdb --args python "$SCRIPT_DIR/recognize.py" $PARAMS
"$SCRIPT_DIR/recognize.py" $PARAMS

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]
then
	echo "Exit status: $EXIT_STATUS" >&2
	exit $EXIT_STATUS
fi

cat "$HYP_FILE"
exit 0
