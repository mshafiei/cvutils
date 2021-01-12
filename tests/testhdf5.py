import h5py
import cv2
fn = '/home/mohammad/Projects/NRV/dataset/12-29-20-globe-point-orth-test-direct/testData/data.hdf5'

f = h5py.File(fn,'r')
for i in range(10):
    outmaskfn = './renderout/outmask-%04d.png' % i
    outfn = './renderout/hdftest-%04d.png' % i
    im = f['in'][i]
    immask = f['in_masks'][i]
    cv2.imwrite(outfn,im[:,:,::-1])
    cv2.imwrite(outmaskfn,immask)
f.close()