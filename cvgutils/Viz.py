import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import cv2
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

def scatter(x,y):
    fig = plt.figure()
    plt.scatter(x,y)
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
    
