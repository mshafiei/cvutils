import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import cv2
import pickle

# from https://stackoverflow.com/questions/7821518/matplotlib-save-plot-to-numpy-array
def get_img_from_fig(fig, dpi=180):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img

def scatter(x):
    """[2d or 3d scatter]

    :param x: [Points [x,y,z]]
    :type x: [list]
    :return: [matplotlib figure]
    :rtype: [pyplot]
    """
    plt.scatter(*x)
    return plt

def plot(x):
    """[2d or 3d plot]

    :param x: [Points [x,y,z]]
    :type x: [list]
    :return: [matplotlib figure]
    :rtype: [pyplot]
    """
    plt.plot(*x)
    return plt

def interpolationSeq(x,y,xs,ys):
    imgs = []
    for x0,y0 in zip(x,y):
        fig = plt.figure()
        plt.scatter(xs,ys,color='b')
        plt.scatter(x0,y0,color='r')
        plotim = get_img_from_fig(fig)
        plt.close()
        imgs.append(plotim)
    return np.stack(imgs,axis=0)
    
def saveInteractiveFig(fn,fig):
    with open(fn,'wb') as fd:
        pickle.dump(fig,fd)

def loadInteractiveFig(fn):
    with open(fn,"rb") as fd:
        return pickle.load(fd)
