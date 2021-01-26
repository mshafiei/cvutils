# cvutils
Computer Vision utilities that I frequently use
Run `conda env create -f environment.yml` to create a conda environment
If you need the mitsuba wrapper you need to install Mitsuba2
I usually need following libraries aside to Anaconda, Cuda and Optix that need to be installed individually:
`sudo apt-get install cmake git libopencv-dev`
Install cuda toolkit by the following,
`sudo apt install nvidia-cuda-toolkit`
this should install cuda toolkit 10.2 as pytorch depends on
command used for compiling mitsuba2 by python37
cmake -GNinja .. -DPYTHON_LIBRARY=/home/mohammad/bin/anaconda3/envs/python37/lib/libpython3.7m.so -DPYTHON_EXECUTABLE=/home/mohammad/bin/anaconda3/envs/python37/bin/python

#next version features
- enable import cvg as a package with cvg.lin, cvg.im, cvg.path, etc. as modules
- create asset directory with
  - Obj files
  - HDR envmaps
  -
