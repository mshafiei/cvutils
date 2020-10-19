import numpy as np
import torch

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
    return v / ((v**2).sum(dim=1) ** 0.5)

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


def perspective(h,w,far,near,fov):
    """[Perspective camera Projection matrix]

    Args:
        far ([float]): [float]
        near ([float]): [float]
        fov ([float]): [float]

    Returns:
        [4x4 array]: [float]
    """
    
    #mitsuba
    offset = [-0.5,-0.5,0.0]

    recip = 1.0 / (far - near)
    tan = np.tan(fov * 0.5)
    cot = 1.0 / tan

    t = torch.eye(4)
    t[0,0] = cot
    t[1,1] = cot
    t[2,2] = far * recip
    t[3,3] = 0.0
    t[2,3] = -near * far * recip
    t[3,2] = 1.0
    

    tinv = torch.eye(4)
    tinv[0,0] = tan
    tinv[1,1] = tan
    tinv[2,2] = 0.0
    tinv[3,3] = 1/near
    tinv[2,3] = 1.0
    tinv[3,2] = (near - far) / (far - near)
    return t, tinv