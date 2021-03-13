from cvgutils.Image import imageseq2avi, loadImageSeq
import glob
import numpy as np
import h5py
import argparse

parser = argparse.ArgumentParser(description='Deploying command')
parser.add_argument('--imfns0',type=str, default='', help='First set filename')
parser.add_argument('--imfns1',type=str, default='', help='Second set filename')
parser.add_argument('--imfns2',type=str, default='', help='Third set filename')
parser.add_argument('--imfns3',type=str, default='', help='Third set filename')
parser.add_argument('--sorted',type=int, default=0, help='Should we sort the files?')
parser.add_argument('--start_id',type=int, default=0, help='Start ID')
parser.add_argument('--end_id',type=int, default=-1, help='End ID')
parser.add_argument('--output',type=str, default='', help='Video filename filename')
args = parser.parse_args()

imfns = []
if(args.imfns0 != ''):
    imfns.append(glob.glob(args.imfns0))

if(args.imfns1 != ''):
    imfns.append(glob.glob(args.imfns1))

if(args.imfns2 != ''):
    imfns.append(glob.glob(args.imfns2))
    
if(args.imfns3 != ''):
    imfns.append(glob.glob(args.imfns3))

imseq = []
if(args.sorted == 1):
    for i in range(len(imfns)):
        imfns[i] = sorted(imfns[i])

for i, fns in enumerate(imfns):
    if(i == 0):
        imseq.append((loadImageSeq(fns[args.start_id:args.end_id])))
    else:
        imseq.append(loadImageSeq(fns[args.start_id:args.end_id]) ** (2.2))
    # print(imseq[i].min(), ' ', imseq[i].max())

im = np.concatenate(imseq,axis=2)
print('shape ', im.shape)
imageseq2avi(args.output,im.transpose(0,3,1,2)[:,::-1,:,:],fps=15)

# # ids = [0,50]
# # colors = np.array([[1.0,1.0,0.2],[1.0,0.1,1.0]])
# ids = [0]
# colors = np.array([[1.0,1.0,1.0]])
# # startid = 60
# # endid = 119
# startid = 0
# endid = 35
# idxs = (np.stack([np.arange(startid,endid)] * len(ids)) + np.stack(ids)[:,None]) % (endid - startid ) + startid
# # imfns0 = '/home/mohammad/Projects/NRV/01-19-21/globe-collocated-lineintegral-sphere/render-video-latest-1/fine_raycolor_[0-9]*.exr'
# imfns1 = '/home/mohammad/Projects/NRV/01-19-21/globe-collocated-lineintegral-sphere/render-video-latest-1/fine_raycolor_pred_[0-9]*.exr'
# # imfns2 = '/home/mohammad/Projects/NRV/01-19-21/globe-collocated-lineintegral-sphere/render-video-latest-1/fine_raycolor_pred_notransmittance*.exr'
# # imfns1 = '/home/mohammad/Projects/NRV/01-19-21/globe-directional-3networks/render-video-latest-1/normal*.exr'
# # imfns2 = '/home/mohammad/Projects/NRV/01-19-21/globe-directional-3networks/render-video-latest-1/roughness*.exr'
# # imfns2 = '/home/mohammad/Projects/NRV/dataset/globe-directions-500/testImgsExr-1bnc/*.png'
# vidfn =  '/home/mohammad/Projects/NRV/01-19-21/globe-collocated-lineintegral-sphere/moving-light.avi'
# gtfn =   '/home/mohammad/Projects/NRV/dataset/globeSaiOrig/testVideo/data.hdf5'

# # imfns0 = glob.glob(imfns0)
# imfns1 = glob.glob(imfns1)
# # imfns2 = glob.glob(imfns2)
# # imfns0 = sorted(imfns0)
# imfns1 = sorted(imfns1)[:70]
# # imfns2 = sorted(imfns2)
# # im0 = loadImageSeq(imfns0) ** (2.2)
# im1 = loadImageSeq(imfns1) ** (2.2)
# # im2 = loadImageSeq(imfns2) ** (2.2)#.astype(np.float32)[:,:,:,:3] / np.iinfo(np.int16).max ) ** (2.2) / 5
# gtims = []
# f = h5py.File(gtfn,'r')
# for i in range(0,0+im1.shape[0]):
#     gtims.append((f['in'][i]))
# f.close()
# gtims = (np.stack(gtims,axis=0).astype(np.float32)[:,:,:,::-1] / 255) ** (2.2)


# im = np.concatenate((im0,im1,im2,gtims),axis=2)
# im = np.concatenate((im1,gtims),axis=2)
# im = np.concatenate((im1,im2),axis=2)
# im = im1
# im = (im[idxs.reshape(-1),:,:,:].reshape(*idxs.shape,*im.shape[1:]) * colors[:,None,None,None,:]).sum(axis=0)
