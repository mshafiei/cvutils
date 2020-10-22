#test perspective matrix vs. mitsuba
import mitsuba
mitsuba.set_variant('scalar_rgb')
from mitsuba.core import Transform4f, xml
import cvgutils.Linalg as lin
import numpy as np
import cvgutils.Mitsuba2XML as mts
import torch
def testCameraMatrix():
    camOrig = torch.Tensor([0.7,0.0,-1.0])
    camLookAt = torch.Tensor([5,0,0])
    camUp = torch.Tensor([0.500,1.0,0.000])
    fov = 60.0
    far = 3
    near = 1
    w = 10
    h = 20
    u, v = np.linspace(0,w-1,w)/w, np.linspace(0,h-1,h)/h
    u, v = np.meshgrid(u,v)
    uv = np.stack((u,v),axis=0).reshape(2,-1)
    # uv = np.array([[0.5,0.5]]).transpose(1,0)
    ext = lin.lookAt(camOrig[None,...],camLookAt[None,...],camUp[None,...])
    cam = mts.camera(camOrig,camLookAt,camUp,fov,ext=ext,near=near,far=far,w=w,h=h,nsamples=4)
    cam = xml.load_string(cam)
    ray = mts.renderRays(cam,uv)
    
    ourRay = lin.sampleRay(h,w,far,near,fov,torch.Tensor(uv)[None,...],ext)
    ourRay = ourRay.reshape(-1,3,h,w).permute(3,2,1,0)[...,0]
    
    print(ray)
    print(ourRay)
    # print(ourRay.reshape(-1,3,h,w).permute(2,3,-1))
    print((ourRay - ray).abs().sum())

def testProjectionMatrix():
    far = 3
    near = 1
    fov = 60.0
    w = 2
    h = 2
    ours = lin.perspective(near,far,fov)
    mts = Transform4f.perspective(fov,near,far)

    diff = (ours - mts).sum()

    assert(diff < 1e-4)


if __name__ == "__main__":
    # testProjectionMatrix()
    testCameraMatrix()