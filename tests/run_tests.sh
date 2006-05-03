#!/bin/sh

scripts="$@"
if [ -z "$scripts" ]; then
    scripts=*.script
fi

for script in $scripts; do
    test=${script%.script}
    echo -n "- $test... "
    sh $script > $test.output
    if cmp $test.output $test.ref >/dev/null; then
	echo "OK"
    else
	echo "FAILED"
    fi
done