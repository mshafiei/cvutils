from cvgutils.Image import imageseq2avi, loadImageSeq
import glob
import numpy as np
import h5py
import argparse
import os 

parser = argparse.ArgumentParser(description='Deploying command')
parser.add_argument('--outdir',type=str, default='../ICCVtex/', help='Directory to store latex file and images')
parser.add_argument('--indir',type=str, default='../ICCVexps/', help='Input directory containing the images')
parser.add_argument('--frameids',type=str, default='127,127,110', help='frameids camma separated')
parser.add_argument('--label',type=str, default='methods', help='Common label')
parser.add_argument('--width',type=float, default=0.3, help='Width of each cell')
args = parser.parse_args()

