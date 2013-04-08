#!/bin/sh

date=`date +"%Y%m%d"`
pkg=decoder-$date.tar.gz

if [ ! -d ../../decoder ]; then
    echo "ERROR: can not find ../../decoder/"
    exit 1
fi

tar -C ../../ -czvf $pkg --exclude-from excludes decoder