import numpy as np
import h5py
import argparse
import os
import cv2
parser = argparse.ArgumentParser(description='Deploying command')
parser.add_argument('--db_fn',type=str, default='/home/mohammad/Projects/NRV/dataset/globeSaiOrig/testVideo/data.hdf5', help='Dataset filename')
parser.add_argument('--entry_name',type=str, default='in', help='Image entry name')
parser.add_argument('--output_dir',type=str, default='/home/mohammad/Projects/NRV/dataset/globeSaiOrig/testVideo/Images', help='Output dir')
args = parser.parse_args()

if(not os.path.exists(args.output_dir)):
    os.makedirs(args.output_dir)

f = h5py.File(args.db_fn,'r')
for i in range(f[args.entry_name].shape[0]):
    outfn = os.path.join(args.output_dir,'%04d.png' % i)
    im = f[args.entry_name][i]
    cv2.imwrite(outfn,im[:,:,::-1])
f.close()