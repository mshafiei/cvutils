import numpy as np
import torch
import mathutils
def pt2xyz(p, t, r = 1):
    """[Polar to cartesian transform]

    Args:
        p ([Tensor bx1x...]): [phi \in [0,2pi]]
        t ([Tensor bx1x...]): [theta \in [0,pi]]
        r (int or Tensor bx1x..., optional): [radius > 0]. Defaults to 1.

    Returns:
        [x,y,z]: [tuple of bx1x... Tensors]
    """

    if(type(p) == torch.Tensor):
        cos = torch.cos
        sin = torch.sin
    else:
        cos = np.cos
        sin = np.sin

    x = r * sin(t) * cos(p)
    y = r * sin(t) * sin(p)
    z = r * cos(t)

    return x,y,z

def xyz2pt(x, y, z):
    """[cartesian to polar transform]

    Args:
        x ([ndarray or tensor]): [description]
        y ([ndarray or tensor]): [description]
        z ([ndarray or tensor]): [description]

    Returns:
        [tuple]: [phi \in [0,2\pi] and theta \in [0, \pi]]
    """
    assert(type(x) == type(y),'type mismatch')
    assert(type(y) == type(z),'type mismatch')

    if(type(x) == torch.Tensor):
        acos = torch.acos
        atan2 = np.atan2
        atan = np.atan
    else:
        acos = np.arccos
        atan2 = np.arctan2
        atan = np.arctan

    r = (x**2+y**2+z**2) ** 0.5
    p = atan2(y,x)
    p[p<0] = p[p<0] + 2*np.pi
    t = acos(z/(r+1e-20))

    return p, t

def pt2uv(p,t):
    """[polar to uniform]

    Args:
        p ([ndarray or tensor]): [description]
        t ([ndarray or tensor]): [description]
    Returns:
        [tuple]: [u \in [0,1] and v \in [0,1]]
    """
    return p/(np.pi * 2),t/np.pi

def uv2pt(u,v):
    """[Converts polar to cartesian]

    Args:
        u ([Tensor or ndarray]): [Azimuth \in [0,1]]
        v ([Tensor or ndarray]): [Elevation \in [0,1]]

    Returns:
        [tuple of Tensor or ndarray]: [\phi \in [0-2 \pi], theta \in [0,\pi] ]
    """
    t = v * np.pi
    p = u * 2 * np.pi
    return p,t

def uv2ptUniform(u,v):
    """[Converts polar to cartesian uniformly on a sphere]

    Args:
        u ([Tensor or ndarray]): [Azimuth \in [0,1]]
        v ([Tensor or ndarray]): [Elevation \in [0,1]]

    Returns:
        [tuple of Tensor or ndarray]: [\phi \in [0-2 \pi], theta \in [0,\pi] ]
    """
    if(type(v) == np.ndarray):
        acos = np.arccos
    elif(type(v) == torch.Tensor):
        acos = torch.acos

    t = acos(1- 2 * v)
    p = u * 2 * np.pi
    return p,t

def naiveSampleOnSphere(u,v):
    p,t = uv2pt(u,v)
    return pt2xyz(p,t)

def uniformSampleOnSphere(u,v):
    p,t = uv2pt(u,v)
    return pt2xyz(p,t)

def vectorNormalize(v):
    """[Normalize a vector as v/||v||]

    Args:
        v ([Tensor b,1,...]): [Input vector]

    Returns:
        [Tensor b,1,...]: [Normalized vector]
    """
    return v / ((v**2).sum(dim=1) ** 0.5 + 1e-20)

def relerr(a,b):
    return (((a-b) ** 2 / b **2) ** 0.5).sum()

def normal2frame(n):
    """[Creates a nxn local frame given a normal vector]

    :param n: [nd Normal vector]
    :type n: [Array]
    :return: [Local frame]
    :rtype: [nxn ndarray]
    """
    r = np.random.rand(len(n)); r / np.linalg.norm(r)
    y = np.cross(n,r)
    x = np.cross(r,y)
    return np.stack((x,y,n),axis=0)
    
def lookAt(Origin, LookAt, Up):

    """[Creates camera matrix in right hand coordinate. Implementation of https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/lookat-function]

    Args:
        Origin ([Tensor b,3,...]): [Camera Origin in world space]
        LookAt ([Tensor b,3,...]): [Camera lookat position in world space]
        Up ([Tensor b,3,...]): [Camera Up direction in world space]

    Returns:
        [Tensor b,4,4,...]: [Camera extrinsic matrix]
    """
    Forward = LookAt - Origin
    Forward = vectorNormalize(Forward)
    Up = vectorNormalize(Up)

    Right = torch.cross(Up, Forward)
    Right = vectorNormalize(Right)
    newUp = torch.cross(Forward, Right)
    newUp = vectorNormalize(newUp)
    m = torch.stack((Right,newUp,Forward),dim=1)
    newOrigin = torch.einsum('bmn...,bn...->bm...', m, Origin)
    m = torch.cat((m,newOrigin.unsqueeze(2)),dim=2)
    m = torch.cat((m,torch.zeros_like(m)[:,0:1,:,...]),dim=1)
    m[:,3,3,...] = 1
    return m



def perspective(fov,near,far):
    """[Generate 4x4 perspective transform matrix. Re-implementation of Mitsuba]

    Args:
        near ([float]): [near plane]
        far ([float]): [far plane]
        fov ([float]): [Field of view in degrees]

    Returns:
        [ndarray]: [4x4 projection matrix]
    """
    #mitsuba
    recip = 1.0 / (far - near)
    cot = 1.0 / np.tan(np.deg2rad(fov * 0.5))

    perspective = torch.diagflat(torch.Tensor([cot,cot,far*recip,0.0]))
    perspective[2,3] = -near * far * recip
    perspective[3,2] = 1.0

    return perspective

def perspective_projection(fov,near,far,filmSize=np.array([1,1]),cropSize=np.array([1,1]),cropOffset=np.array([0,0])):
    """[Reimplementation of Mitsuba perspective_projection function]

    Args:
        fov ([float]): [Field of view in degrees]
        near ([float]): [Near plane]
        far ([float]): [Far plane]
        filmSize ([1x2 ndarray], optional): [Film size]. Defaults to np.array([1,1]).
        cropSize ([1x2 ndarray], optional): [crop size]. Defaults to np.array([1,1]).
        cropOffset ([1x2 ndarray], optional): [Crop offset]. Defaults to np.array([0,0]).

    Returns:
        [4x4 tensor]: [Perspective camera projection matrix]
    """

    aspect = filmSize[0] / filmSize[1]
    rel_offset = cropOffset / filmSize
    rel_size = cropSize / filmSize
    p = perspective(fov,near,far)

    translate = torch.eye(4)
    translate[:3,-1] = torch.Tensor([-1.0, -1.0 / aspect, 0.0])
    scale = torch.diagflat(torch.Tensor([-0.5,-0.5*aspect,1.0,1.0]))
    translateCrop = torch.eye(4)
    translateCrop[:3,-1] = torch.Tensor([-rel_offset[0],-rel_offset[1],0.0])
    scaleCrop = torch.diagflat(torch.Tensor([1/rel_size[0],1/rel_size[1],1.0,1.0]))
    m1 = torch.mm(scaleCrop,torch.mm(translateCrop,torch.mm(scale,torch.mm(translate,p))))
    return m1

def sampleRay(h,w,far,near,fov,samples,ext):
    """[Reimplementation of Mitsuba sample_ray]

    Args:
        far ([float]): [float]
        near ([float]): [float]
        fov ([float]): [float]

    Returns:
        [4x4 array]: [float]
    """
    
    aspect = w/h
    
    camera_to_sample = perspective_projection(fov,near,far,filmSize=np.array([w,h]),cropSize=np.array([w,h]))
    sample_to_camera = torch.inverse(camera_to_sample[None,...])
    samples = np.concatenate((samples,np.zeros_like(samples[:,0:1,...]),np.ones_like(samples[:,0:1,...])),axis=1)
    
    d = torch.einsum('abc,acd->abd',sample_to_camera,torch.Tensor(samples))
    d = d[:,:3,...] / d[:,3,...]
    d = vectorNormalize(d)
    d = torch.einsum('abc,ac...->ab...',ext[:,:3,:3,...].transpose(2,1),d[:,:3,...])
    
    return d

def u1tou2(u1,u2):
    """[Compute shortest rotation between two vectors]

    Args:
        u1 ([ndarray]): [1st vector]
        u2 ([ndarray]): [2nd vector]

    Returns:
        [Quaternion]: [Shortest rotation]
    """
    u1 = u1 / np.sum(u1 ** 2) ** 0.5
    u2 = u2 / np.sum(u2 ** 2) ** 0.5
    a = np.cross(u1,u2)
    t = np.arccos(np.sum(u1 * u2))
    q = mathutils.Quaternion(a,t)
    print('u1 ', u1, ' u2 ',u2, ' a ', a , ' t ', t , ' q ', q)
    exit(1)
    return q
    