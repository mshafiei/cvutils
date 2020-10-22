import cvgutils.Mitsuba2XML as mts
import cvgutils.Linalg as lin
import cvgutils.Dir as dr
import numpy as np
import torch
if __name__ == "__main__":
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
    camOrig = torch.Tensor([0.0,0.0,-1.0])
    # camLookAt = torch.Tensor([0.0,0.0,0.0])
    camLookAt = torch.Tensor([0,0,0])
    camUp = torch.Tensor([0.000,1.0,0.000])

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
    near = 1
    far = 3
    fov = 60.0
    w = 2
    h = 2
    

    material = mts.diffuse(diffuseReflectance)
    shape = mts.sphere(center, radius,material)
    light = mts.pointlight(camOrig[None,...],intensity)

    ext = lin.lookAt(camOrig[None,...],camLookAt[None,...],camUp[None,...])
    camera = mts.camera(camOrig,camLookAt,camUp,fov,ext=ext,near=near,far=far,w=w,h=h,nsamples=nsamples)
    scene = mts.generateScene(shape,light,camera)

    u,v = np.linspace(0,1,h),np.linspace(0,1,w)
    u,v = np.meshgrid(u*0+0.5,v*0+0.5)
    rays = mts.renderRays(scene.sensors()[0])
    d = lin.perspective(h,w,far,near,fov,np.stack((u,v),axis=-1).reshape(-1,2),ext)
    d = d.transpose(1,0).reshape(*rays.shape)
    print(rays)
    
#sample rays by your implementation
#compare