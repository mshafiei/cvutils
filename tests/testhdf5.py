import h5py
import cv2
fn = '/home/mohammad/Projects/NRV/dataset/sceneBlender/testData/data.hdf5'
outfn = './renderout/hdftest.png'
outmaskfn = './renderout/hdftest/outmask.png'
f = h5py.File(fn,'r')
im = f['in'][0]
immask = f['in_masks'][0]
cv2.imwrite(outfn,im[:,:,::-1])
cv2.imwrite(outmaskfn,immask)
f.close()