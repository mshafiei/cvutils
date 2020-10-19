import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import cvgutils.Linalg as lin
import os
ntheta = 5
nphi = 1
theta = np.linspace(0.1, np.pi-0.1, ntheta)
phi = np.linspace(0, np.pi * 2, nphi)
theta, phi = np.meshgrid(theta,phi)
theta, phi = (theta.reshape(-1), phi.reshape(-1))
x,y,z = lin.pt2xyz(phi,theta)
x,y,z = [y,z,-x]

fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x,y,z)
ax.view_init(elev=45,azim=45)
if(not os.path.exists('renderout')):
    os.makedirs('renderout')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.savefig("renderout/fig1.png")