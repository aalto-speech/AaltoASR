#!/bin/sh
#
# Good for recognizing a couple of files specified in the command line. Writes
# LNAs to
#   /share/work/<username>/recognitions/<AM options>,
# and possibly lattices to
#   /share/work/<username>/recognitions/<AM options>/<decoder options>.
# Writes the recognitions results (hypotheses) under
#   /share/work/<username>/results
# and displays on the screen.

SCRIPT_DIR=$(dirname $0)

if [ "$AM" = "" ]
then
	echo "AM environment variable needs to be specified." >&2
	exit 2
fi
if [ "$LM" = "" ]
then
	echo "LM environment variable needs to be specified." >&2
	exit 2
fi
if [ "$DICTIONARY" = "" ]
then
	echo "DICTIONARY environment variable needs to be specified." >&2
	exit 2
fi
if [ "$BEAM" = "" ]
then
	BEAM=280
fi
if [ "$LM_SCALE" = "" ]
then
	LM_SCALE=30
fi
if [ "$TOKEN_LIMIT" = "" ]
then
	TOKEN_LIMIT=100000
fi
if [ "$RECOGNITIONS_DIR" = "" ]
then
	RECOGNITIONS_DIR="/share/work/$USER/recognitions"
fi
if [ "$RESULTS_DIR" = "" ]
then
	RESULTS_DIR="/share/work/$USER/results"
fi

AM_OPT=$(basename $AM)

DECODER_OPT=""
if [ "$DECODER_VER" != "" ]
then
	DECODER_OPT="$DECODER_VER--"
fi
DECODER_OPT="$DECODER_OPT$(basename $LM)"

PARAMS=""

if [ "$AKU" != "" ]
then
	PARAMS="$PARAMS --aku $AKU"
fi
if [ "$DECODER" != "" ]
then
	PARAMS="$PARAMS --decoder $DECODER"
fi

# Specify binary LM if .bin or .fsabin file exists.
if [ "$FSA" != "" ]
then
	PARAMS="$PARAMS --fsa"
	if [ -e "$LM.fsabin" ]
	then
		PARAMS="$PARAMS --bin-lm $LM.fsabin"
	fi
	DECODER_OPT=$DECODER_OPT-fsa
else
	if [ -e "$LM.bin" ]
	then
		PARAMS="$PARAMS --bin-lm $LM.bin"
	fi
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
        DECODER_OPT="$DECODER_OPT-$(basename $CLASSES .classes)-"
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
fi

if [ "$HESITATIONS" != "" ]
then
	PARAMS="$PARAMS --hesitations"
	DECODER_OPT=$DECODER_OPT-oh2
fi

DECODER_OPT=$DECODER_OPT-b$BEAM-s$LM_SCALE
DECODER_OPT=$DECODER_OPT-tl$(echo $TOKEN_LIMIT|"$SCRIPT_DIR/human-readable.awk")

if [ "$ADAPTATION" != "" ]
then
	PARAMS="$PARAMS --adapt $ADAPTATION"
	DECODER_OPT="$DECODER_OPT-$ADAPTATION"
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
