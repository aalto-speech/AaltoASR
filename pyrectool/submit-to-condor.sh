#!/bin/bash

interrupt_handler() {
	condor_rm -all
	rm -f $jobdesc
	exit 3
}

logfile="/share/work/$USER/log/condor.log"

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

#
# Write a Condor job description file.
#

wrapper="$(dirname $0)/exec-line.sh"
queuesize=$(wc -l $script | awk '{print $1}')
myname=$(basename $0)
jobdesc=$(mktemp "/tmp/$myname.XXXXXXXXXX")

echo "requirements = ICSLinux == 2012" >>"$jobdesc"
echo "executable = $wrapper" >>"$jobdesc"
echo "arguments = $script \$(Process)" >>"$jobdesc"

if [ "$logfile" != "" ]; then
	echo "log = $logfile" >>"$jobdesc"
	echo "output = $logfile.out.\$(Process)" >>"$jobdesc"
	echo "error = $logfile.err.\$(Process)" >>"$jobdesc"
	
	rm -f "$logfile" "$logfile.out."* "$logfile.err."*
	mkdir -p $(dirname $logfile)
	touch "$logfile"
	for (( process = 1; process < queuesize; process++ ))
	do
		touch "$logfile.out.$process"
		touch "$logfile.err.$process"
	done
fi

echo "queue $queuesize" >>"$jobdesc"

condor_submit $jobdesc
rv=$?
rm -f $jobdesc

if [ $rv -eq 0 ]
then
	trap 'interrupt_handler' INT
	
	tail -f $logfile.out.* $logfile.err.* &
	
	echo "Waiting for jobs to finish."
	condor_wait "$logfile"
	
	# Kill the tail -f background job.
	kill $!
	
	trap - INT
else
	exit $rv
fi
