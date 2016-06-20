#!/usr/bin/env python3

from rectool import RecognizerToolbox

rt = RecognizerToolbox()
if not rt.use_existing_lnas():
	if not rt.use_existing_adaptation():
		rt.generate_lnas()
		rt.decode_state_segmentations()
		rt.estimate_adaptation()
	rt.generate_lnas()
rt.decode_batch()
