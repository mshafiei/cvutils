import h5py
import numpy as np
import cv2
import pickle
import cvgutils.Dir as Dir
import os
from inspect import currentframe, getframeinfo
def images2hdf5(fn,imgs,masks):

    f = h5py.File(fn, "w")
    f.create_dataset("in", imgs.shape, dtype = np.uint8,chunks=(1,1,*imgs.shape[2:]),compression="lzf", shuffle=True)
    f.create_dataset("in_masks", masks.shape, dtype = np.uint8,chunks=(1,1,*masks.shape[2:]),compression="lzf", shuffle=True)
    d = f["in"]
    for i, im in enumerate(imgs):
        d[i] = imgs[i]

    d2 = f["in_masks"]
    for i, im in enumerate(masks):
        d2[i] = masks[i]
    f.close()

def savePickle(relfn,obj):
    #create parent path if not exist
    fn = os.path.abspath(relfn)
    parentPath = Dir.getPathFilename(fn)
    Dir.createIfNExist(parentPath)
    with open(fn,'wb') as fd:
        pickle.dump(obj,fd)

def loadPickle(fn):
    with open(fn,'rb') as fd:
        obj = pickle.load(fd)
    return obj

def printLine(frame,debugTiming):
    #use case:
#     import inspect
#     frame = inspect.currentframe()
    if(debugTiming):
        frameinfo = getframeinfo(frame)
        print('line: ',frameinfo.filename, frameinfo.lineno)