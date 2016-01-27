#!/bin/bash -e

interrupt_handler() {
	scancel $(printf " %s" "${jobs[@]}") 2>/dev/null
	rm -f "$logfile".out.* "$logfile".err.*
	rmdir --ignore-fail-on-non-empty $(dirname "$logfile")
	exit 3
}

if [ -d "/triton/ics/work/$USER" ]
then
	logfile="/triton/ics/work/$USER/log/$(date '+%Y-%m-%d-%H%M%S')/rectool.log"
elif [ -d "/triton/elec/work/$USER" ]
then
	logfile="/triton/elec/work/$USER/log/$(date '+%Y-%m-%d-%H%M%S')/rectool.log"
else
	logfile=
fi

while [ $# -gt 1 ]
do
	case "$1" in
	-log)
		logfile=$2;
		shift; shift;;
	-f)
		script=$2;
		shift; shift;;
	-J)
		njobs=$2;
		shift; shift;;
	-*)
		echo "unknown $1 option "
		exit 1;;
	*)
		break;;
	esac
done

wrapper="$(dirname $0)/srun-wrapper.sh"
execline="$(readlink -f $(dirname $0))/exec-line.sh"
queuesize=$(wc -l $script | awk '{print $1}')

if [ "$logfile" != "" ]; then
	mkdir -p $(dirname "$logfile")
	rm -f "$logfile".out.* "$logfile".err.*
fi

for i in $(seq 0 $((queuesize-1)))
do
	command="sbatch --partition=short --qos=short --time=4:00:00 --mem=10G"
	if [ "$SLURM_EXCLUDE_NODES" != "" ]; then
		command="$command --exclude=$SLURM_EXCLUDE_NODES"
	fi
	if [ "$logfile" != "" ]; then
		touch "$logfile".out.$i
		touch "$logfile".err.$i
		command="$command -o $logfile.out.$i -e $logfile.err.$i"
	fi
	command="$command $execline $script $i"
	output=$($command)
	rv=$?
	if [ $rv -ne 0 ]
	then
		echo "Command returned a non-zero exit code: $command"
		echo "Output: $output"
		echo "Exit code: $rv"
		scancel $jobs
		rm -f "$logfile".out.* "$logfile".err.*
		exit $rv
	fi
	
	jobid=$(echo $output | sed -r 's/^Submitted batch job ([0-9]+)$/\1/')
	jobs[$i]=$jobid
done

trap 'interrupt_handler' INT

tail -f "$logfile".out.* "$logfile".err.* &

echo "Waiting for job to finish."

while output=$(squeue --jobs=$(printf ",%s" "${jobs[@]}") --noheader)
do
	if [ -z "$output" ]
	then
		break
	fi
	sleep 4
done

failed=$(sacct --format=jobid,state --parsable2 --jobs=$(printf ",%s" "${jobs[@]}") --noheader | awk -F'|' '{ if ($2 != "COMPLETED") print; }')
if [ "$failed" != "" ]; then
	echo "Some jobs failed:"
	echo "$failed"
	echo "See log files for details:"
	echo "$logfile.out.*"
	echo "$logfile.err.*"
	exit 2
fi

# Kill the tail -f background job and delete the logs.
kill $!
rm -f "$logfile".out.* "$logfile".err.*
rmdir --ignore-fail-on-non-empty $(dirname "$logfile")

trap - INT
