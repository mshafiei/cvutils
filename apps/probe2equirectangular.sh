#!/bin/bash
probe_fn='/home/mohammad/Projects/NRV/NrArtFree/cvgutils/tests/testimages/grace_probe.hdr'
eq_fn='/home/mohammad/Projects/NRV/NrArtFree/cvgutils/tests/testimages/grace_eq.exr'

python probe2equirectangular.py --probe_fn probe_fn --eq_fn eq_fn