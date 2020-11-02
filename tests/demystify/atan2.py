import numpy as np
import cvgutils.Viz as viz
import matplotlib.pyplot as plt
import cv2
import cvgutils.Linalg as lin
if __name__ == "__main__":
    p = np.linspace(0,2*np.pi,100)
    x = np.cos(p)
    y = np.sin(p)
    f,t = lin.xyz2pt(x,y,0)
    # f = np.arctan2(y,x)
    # f[f<0] += 2*np.pi
    im = viz.scatter(p,f)
    im = viz.get_img_from_fig(im)
    plt.close()
    imxy = viz.scatter(x,y)
    imxy = viz.get_img_from_fig(imxy)
    cv2.imwrite('renderout/atan.png',np.concatenate((im,imxy),axis=0))
    