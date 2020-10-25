import h5py
import cv2
fn = '/home/mohammad/Projects/NRV/dataset/bunny-5/trainData/data.hdf5'
outfn = './out.png'
f = h5py.File(fn,'r')
im = f['in'][0]
cv2.imwrite(outfn,im)
f.close()