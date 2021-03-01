#!/bin/bash
imseq='/home/mohammad/Projects/NRV/02-23-21/globe-randsamples-nograd/render-video-latest-1/shadow_volume_%04d_.exr'
indexfn='/home/mohammad/Projects/NRV/dataset/globe-envmap-discrete/testData/index.pickle'
envmap='/home/mohammad/Projects/NRV/NrArtFree/cvgutils/tests/testimages/grace_eq.exr'
output='/home/mohammad/Projects/NRV/02-23-21/globe-randsamples-nograd/grace.exr'
python pointlight2envmap.py --imseq $imseq --indexfn $indexfn --envmap $envmap --output $output