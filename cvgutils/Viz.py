import numpy as np
import torch
import matplotlib.pyplot as plt
import io
import cv2
import pickle
import wandb
import cvgutils.Dir as Dir
from tensorboardX import SummaryWriter
import os

class logger:
    def __init__(self,path,ltype,projectName,expName):
        self.path = os.path.join(path,expName)
        if(ltype == 'tb' or ltype == 'filesystem'):
            Dir.createIfNExist(self.path)
        self.ltype = ltype
        self.step = 0
        if(self.ltype == 'wandb'):
            wandb.init(project=projectName,name=expName)
        elif(self.ltype == 'tb'):
            self.writer = SummaryWriter(self.path)
        # elif(self.ltype == 'filesystem'):



    def addImage(self,im,label):
        if(self.ltype == 'wandb'):
            wandb.log({label:[wandb.Image(im)]},step=self.step)
        elif(self.ltype == 'tb'):
            imshow = im
            if(type(im) == np.ndarray):
                if(im.dtype == np.uint8):
                    imshow = torch.Tensor(im).permute(2,0,1) / 255
            self.writer.add_image(label.replace(' ','_'), imshow, self.step)
        elif(self.ltype == 'filesystem'):
            name = os.path.join(self.path,'%010i_%s.png' %(self.step, label.replace(' ','_')))
            cv2.imwrite(name,im)

    def addLoss(self,loss,label):
        if(self.ltype == 'wandb'):
            wandb.log({label:loss},self.step)
        elif(self.ltype == 'tb'):
            self.writer.add_scalar(label, float(loss), self.step)



    def takeStep(self):
        self.step += 1

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

def plot(x,y,marker='.',xlabel='x',ylabel='y',title='',step=None,logger=None,ptype='plot'):
    
    fig, ax = plt.subplots()
    if(ptype=='plot'):
        ax.plot(x,y)
        ax.scatter(x,y,marker=marker)
    elif(ptype=='scatter'):
        ax.scatter(x,y)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    im = get_img_from_fig(fig)
    plt.close(fig)

    try:
        if((not (logger is None)) and (not (step is None))):
            logger.addImage(im,title,step)
    except Exception as e:
        print(e)
    
    return im
def plotOverlay(x,y1,y2,marker='.',xlabel='x',ylabel='y',legend=['Prediction','GT'],title='',step=None,logger=None,ptype='plot'):
    
    fig, ax = plt.subplots()
    if(ptype=='plot'):
        ax.plot(x,y1,'r')
        ax.plot(x,y2,'b')
        ax.scatter(x,y1,marker=marker,color='r')
        ax.scatter(x,y2,marker=marker,color='b')
    elif(ptype=='scatter'):
        ax.scatter(x,y1,marker=marker,color='r')
        ax.scatter(x,y2,marker=marker,color='b')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(legend)
    im = get_img_from_fig(fig)
    plt.close(fig)

    try:
        if((not (logger is None)) and (not (step is None))):
            logger.addImage(im,title,step)
    except Exception as e:
        print(e)
    
    return im

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
