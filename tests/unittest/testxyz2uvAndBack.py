import numpy as np
import cvgutils.Linalg as lin
import cvgutils.Image as im

if __name__ == "__main__":
    width = 512
    height = 512
    u = np.random.rand(width)
    v = np.random.rand(height)
    u,v = np.meshgrid(u,v)
    p,t = lin.uv2pt(u,v)
    x,y,z = lin.pt2xyz(p,t)
    pb,tb = lin.xyz2pt(x,y,z)
    ub,vb = lin.pt2uv(p,t)

    assert((((pb - p) ** 2) ** 0.5).sum() < 1e-5)
    assert((((tb - t) ** 2) ** 0.5).sum() < 1e-5)
    assert((((ub - u) ** 2) ** 0.5).sum() < 1e-5)
    assert((((vb - v) ** 2) ** 0.5).sum() < 1e-5)
