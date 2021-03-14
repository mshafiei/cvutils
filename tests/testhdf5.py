import h5py
import cv2
import numpy as np
# fn = '/home/mohammad/Projects/NRV/dataset/12-29-20-globe-point-orth-test-direct/testData/data.hdf5'
# fn = '/home/mohammad/Projects/NRV/dataset/globeSaiOrig/testVideo/data.hdf5'
fn = '/home/mohammad/Projects/NRV/dataset/buddha/testData-pointlight/data.hdf5'
f = h5py.File(fn,'r')
for i in range(0,400):
    # outmaskfn = './renderout/outmask-%04d.png' % i
    outfn = '/home/mohammad/Projects/NRV/dataset/buddha/testImgsExr-pointlight/%07d.png' % i
    im = f['in'][i]
    # immask = f['in_masks'][i]
    # cv2.imwrite(outfn,((im[:,:,::-1]/255.0) ** 2.2).astype(np.float32))
    cv2.imwrite(outfn,im[:,:,::-1])
    # cv2.imwrite(outmaskfn,immask)
f.close()