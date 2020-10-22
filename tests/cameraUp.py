#This script renders input data for Deep Reflectance Volume
import cvgutils.Mitsuba2XML as mts
import cvgutils.Image as im
import cvgutils.Linalg as lin
import cvgutils.Dir as dr
import cvgutils.Utils as util
import cv2
import numpy as np
import torch
import os
from scipy.interpolate import interp1d


def randomPathSphere(x):
        """[Given parameers of a 1d path returns phi, theta on that path]

        Args:
            x ([ndarray]): [path parameters]
        Returns:
            [tuple]: [points on the path on a sphere]
        """
        u,v = np.random.rand(2,len(x))
        return lin.uv2ptUniform(u,v)

def randomBicubicPathSphere(x,r=1,nods=5):
    """[Given parameers of a 1d path returns phi, theta on that path]

    Args:
        x ([ndarray]): [path parameters]
        r (float, optional): [Raidus of the sphere]. Defaults to 1.
        nods (int, optional): [Number of nods for spline]. Defaults to 3.

    Returns:
        [tuple]: [points on the path on a sphere]
    """
    x0, y0 = np.random.rand(2,nods)
    f = interp1d(x0, y0, kind='cubic')
    pathRange = (x0.max() - x0.min())
    newRange = (x.max() - x.min())
    u =  x / newRange * pathRange + x0.min()
    v = np.clip(f(u),0,1)
    return lin.uv2ptUniform(u,v)
    
    
def dumpCameraInfo(trajectory,shape,maskShape, fov, camLookAt,camUp, outdir,nsamples):
    ps, ts = trajectory
    images = []
    for i, (t, p) in enumerate(zip(ts.reshape(-1),ps.reshape(-1))):

        x,y,z = lin.pt2xyz(p,t,r)
        xl,yl,zl = [x,y,z]
        near = (x**2 + y**2 +z**2) ** 0.5 - 1.0
        far = (x**2 + y**2 +z**2) ** 0.5 + 1.0
        light = mts.pointlight([xl,yl,zl],intensity)
        
        ext = lin.lookAt(torch.Tensor([[x,y,z]]),camLookAt[None,...],camUp[None,...])
        camera = mts.camera([x,y,z],camLookAt,camUp,fov,ext=ext,near=near,far=far,w=w,h=h,nsamples=nsamples)
        scene = mts.generateScene(shape,light,camera)
        img = mts.renderScene(scene)
        images.append(img)

    images = np.stack(images,axis=0)
    im.imageseq2avi('renderout/tst.avi',images.transpose(0,3,1,2),10)

if __name__ == "__main__":
    #TODO: create math module
    #TODO: create LatLong related modules
    outdir = '/home/mohammad/Projects/NRV/dataset/simple/trainData'
    outdirTest ='/home/mohammad/Projects/NRV/dataset/simple/testData'
    outfmt = '%04d-%04d.png'
    outfmtmask = 'mask-%04d-%04d.png'
    
    texturefn = 'cvgutils/tests/testimages/5x5pattern.png'
    objfn = 'cvgutils/tests/testobjs/z.obj'
    dr.createIfNExist(outdir)
    dr.createIfNExist(outdirTest)
    center = [0,0,0]
    intensity = [1.0,1.0,1.0]
    nsamples = 15
    radius = 0.8
    diffuseReflectanceMask = [1.0,1.0,1.0]
    specularReflectance = [1.0,1.0,1.0]
    diffuseReflectance = texturefn
    dxs = torch.Tensor([0.0,1.0,0.0]) * 0.2
    camOrig = torch.Tensor([1.0,0.0,0.0])
    # camLookAt = torch.Tensor([0.0,0.0,0.0])
    camLookAt = torch.Tensor([0,0,0])
    camUp = torch.Tensor([0.0001,0.0,1.000])

    ntheta = 10
    nphi = 10
    nthetal = 15
    nphil = 15

    r = 1.8
    rl = 1.8
    intior = 1.0
    extior = 1.000277
    k = 0
    alpha = 0.0
    fov = 60.0
    w = 64
    h = 64
    

    x = np.linspace(0,1,ntheta*nphi)
    
    u,v = np.linspace(0,1,15), np.linspace(0.1,0.9,15)
    ps,ts = lin.uv2ptUniform(u,v)
    ps,ts = np.meshgrid(ps,ts)
    # ps, ts = randomPathSphere(x)
    trainTraj = [ps,ts]

    material = mts.diffuse(diffuseReflectanceMask)
    maskShape = mts.sphere(center, radius,material)
    material = mts.diffuse(diffuseReflectance)
    shape = mts.sphere(center, radius,material)
    
    dumpCameraInfo(trainTraj,shape,maskShape, fov, camLookAt,camUp, outdir,nsamples)

    