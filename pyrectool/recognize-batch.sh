#!/bin/sh
#
# Recognizes a batch, writing LNAs to
#   /share/work/<username>/recognitions/<AM options>,
# and possibly lattices to
#   /share/work/<username>/recognitions/<AM options>/<decoder options>,
# and writes the recognitions results (hypotheses) under
#   /share/work/<username>/results
# with a name unique to selected options.

SCRIPT_DIR=$(dirname $0)
SORT_TRN="$SCRIPT_DIR/../scoring-scripts/sort-trn.py"

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
if [ "$NUM_BATCHES" = "" ]
then
	NUM_BATCHES=1
fi
if [ "$MAX_PARALLEL" = "" ]
then
	MAX_PARALLEL=1
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
	RECOGNITIONS_DIR="$HOME/recognitions"
fi
if [ "$RESULTS_DIR" = "" ]
then
	RESULTS_DIR="$HOME/results"
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
HYP_FILE="$RESULTS_DIR/$AM_OPT--$DECODER_OPT.trn"

mkdir -p "$REC_DIR"
mkdir -p "$RESULTS_DIR"

PARAMS="$PARAMS --am $AM"
PARAMS="$PARAMS --dictionary $DICTIONARY.lex"
PARAMS="$PARAMS --beam $BEAM"
PARAMS="$PARAMS --language-model-scale $LM_SCALE"
PARAMS="$PARAMS --token-limit $TOKEN_LIMIT"
PARAMS="$PARAMS --hypothesis-file $HYP_FILE"
PARAMS="$PARAMS --work-directory $WORK_DIR"
PARAMS="$PARAMS --rec-directory $REC_DIR"
PARAMS="$PARAMS -f $AUDIO_LIST"

if [ $NUM_BATCHES -gt 1 ]
then
	"$SCRIPT_DIR/recognize-parallel.py" $PARAMS -B $NUM_BATCHES -P $MAX_PARALLEL
else
	"$SCRIPT_DIR/recognize.py" $PARAMS
fi

EXIT_STATUS=$?
if [ $EXIT_STATUS -ne 0 ]
then
	echo "Exit status: $EXIT_STATUS" >&2
	exit $EXIT_STATUS
fi

if [ -x "$SORT_TRN" ]
then
	"$SORT_TRN" "$HYP_FILE"
fi

echo "Wrote $HYP_FILE."
exit 0
