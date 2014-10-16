#!/usr/bin/env python

from rectool import RecognizerToolbox
import sys

rt = RecognizerToolbox()
if rt.use_existing_state_segmentations():
	sys.exit(0)

if not rt.use_existing_lnas():
	rt.generate_lnas()

rt.decode_state_segmentations()
