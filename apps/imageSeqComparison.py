from cvgutils.Image import imageseq2avi, loadImageSeq
import glob
import numpy as np
import h5py
import argparse
import os
import torch
import cv2

parser = argparse.ArgumentParser(description='Deploying command')
parser.add_argument('--gtfns',type=str, default='', help='First set filename')
parser.add_argument('--imfns1',type=str, default='', help='Second set filename')
parser.add_argument('--imfns2',type=str, default='', help='Third set filename')
parser.add_argument('--imfns3',type=str, default='', help='Third set filename')
parser.add_argument('--imfns4',type=str, default='', help='Fourth set filename')
parser.add_argument('--sorted',type=int, default=0, help='Should we sort the files?')
parser.add_argument('--start_id',type=int, default=0, help='Start ID')
parser.add_argument('--end_id',type=int, default=-1, help='End ID')
parser.add_argument('--output',type=str, default='', help='Output text filename')
args = parser.parse_args()
torch.save(args,'args.pickle')
# exit(0)
# args = torch.load('args.pickle')
imfns = []
if(args.gtfns != ''):
    imfns.append(glob.glob(args.gtfns))

if(args.imfns1 != ''):
    imfns.append(glob.glob(args.imfns1))

if(args.imfns2 != ''):
    imfns.append(glob.glob(args.imfns2))
    
if(args.imfns3 != ''):
    imfns.append(glob.glob(args.imfns3))

if(args.imfns4 != ''):
    imfns.append(glob.glob(args.imfns4))

imseq = []
if(args.sorted == 1):
    for i in range(len(imfns)):
        imfns[i] = sorted(imfns[i])

def MSE(x,gt):
    """[Compute scale invariant mean square error]

    Args:
        x ([ndarray or Tensor]): [n x b where n is number of observations and b is number of pixels]
        gt ([ndarray or Tensor]): [n x b where n is number of observations and b is number of pixels]

    Returns:
        [float]: [scale invariant MSE]
    """
    return (x - gt) ** 2

def scaleInvariantMSE(x,gt):
    """[Compute scale invariant mean square error]

    Args:
        x ([ndarray or Tensor]): [n x b where n is number of observations and b is number of pixels]
        gt ([ndarray or Tensor]): [n x b where n is number of observations and b is number of pixels]

    Returns:
        [float]: [scale invariant MSE]
    """
    a = ((x * gt).sum(-1,keepdims=True) / (x ** 2).sum(-1,keepdims=True))
    return (a*x - gt) ** 2

def relativeMSE(x,gt):
    """[Compute relative mean square error]

    Args:
        x ([ndarray or Tensor]): [n x b where n is number of observations and b is number of pixels]
        gt ([ndarray or Tensor]): [n x b where n is number of observations and b is number of pixels]

    Returns:
        [float]: [relative MSE]
    """
    a = ((x * gt).sum(-1,keepdims=True) / (x ** 2).sum(-1,keepdims=True))
    diff = (((a*x - gt) ** 2).sum() / (gt ** 2).sum()) ** 0.5
    errs = diff
    return errs

MSEerrs = []
relMSEerrs = []
siMSEerrs = []
for i, fns in enumerate(imfns):
    if(i == 0 or i == 1):
        imseq.append((loadImageSeq(fns[args.start_id:args.end_id])))
    else:
        imseq.append(loadImageSeq(fns[args.start_id:args.end_id]) ** (2.2))
    if(i > 0):
        x = imseq[i].reshape(imseq[i].shape[0],-1)
        y = imseq[0].reshape(x.shape[0],-1)
        mse = MSE(x,y)
        relative = relativeMSE(x,y)
        siMSE = scaleInvariantMSE(x,y)
        path = os.path.abspath(os.path.join(fns[0],os.pardir))
        # for j, (im,pred,gt) in enumerate(zip(mse,imseq[i],imseq[0])):
        #     # errpath = os.path.join(path,'error-mse-%04d.exr' % j)
        #     # cv2.imwrite(errpath, im.reshape(*imseq[i].shape[1:]))
        #     errpath = os.path.join(path,'error-mse-%04d.png' % j)
        #     im = np.concatenate((gt,pred,im.reshape(*imseq[i].shape[1:])),axis=1) ** (1/2.2) * 255
        #     cv2.imwrite(errpath, im.astype(np.uint8))
        # for j, (im,pred,gt) in enumerate(zip(siMSE,imseq[i],imseq[0])):
        #     # errpath = os.path.join(path,'error-simse-%04d.exr' % j)
        #     # cv2.imwrite(errpath, im.reshape(*imseq[i].shape[1:]))
        #     errpath = os.path.join(path,'error-simse-%04d.png' % j)
        #     im = np.concatenate((gt,pred,im.reshape(*imseq[i].shape[1:])),axis=1) ** (1/2.2) * 255
        #     cv2.imwrite(errpath, im.astype(np.uint8))


        MSEerrs.append(mse.mean())
        relMSEerrs.append(relative.mean())
        siMSEerrs.append(siMSE.mean())

np.savetxt(args.output.split('.txt')[0] + '-MSE.txt',MSEerrs)
np.savetxt(args.output.split('.txt')[0] + '-siRelMSE.txt',relMSEerrs)
np.savetxt(args.output.split('.txt')[0] + '-siMSE.txt',siMSEerrs)