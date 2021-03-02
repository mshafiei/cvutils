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
import scipy

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
            cv2.imwrite(name,im[...,::-1])

    def addScalar(self,scalar,label):
        fn = os.path.join(self.path,label + '.txt')
        if(self.ltype == 'wandb'):
            wandb.log({label:scalar},self.step)
        elif(self.ltype == 'tb'):
            self.writer.add_scalar(label, float(scalar), self.step)
        elif(self.ltype == 'filesystem'):
            with open(fn, 'a') as fd:
                fd.write('%s : %f' % (label,scalar) )

    def addString(self,text,label):
        fn = os.path.join(self.path,label + '.txt')
        if(self.ltype == 'filesystem'):
            with open(fn, 'a') as fd:
                fd.write(text + '\n')



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

def plot(x,y,marker='.',xlabel='x',ylabel='y',title='',step=None,logger=None,xlim=None,ylim=None,ptype='plot'):
    
    fig, ax = plt.subplots()
    if(ptype=='plot'):
        ax.plot(x,y)
        ax.scatter(x,y,marker=marker)
    elif(ptype=='scatter'):
        ax.scatter(x,y)
    if(xlim is not None):
        plt.xlim(xlim[0],xlim[1])
    if(ylim is not None):
        plt.ylim(ylim[0],ylim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    im = get_img_from_fig(fig)
    plt.close('all')

    try:
        if((not (logger is None)) and (not (step is None))):
            logger.addImage(im,title,step)
    except Exception as e:
        print(e)
    
    return im

def plot3(x,y,z,marker='.',xlabel='x',ylabel='y',zlabel='z',title='',step=None,logger=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # u = np.linspace(0, np.pi, 30)
    # v = np.linspace(0, 2 * np.pi, 30)

    # x = np.concatenate((x,np.outer(np.sin(u), np.sin(v)).reshape(-1)),axis=-1)
    # y = np.concatenate((y,np.outer(np.sin(u), np.cos(v)).reshape(-1)),axis=-1)
    # z = np.concatenate((z,np.outer(np.cos(u), np.ones_like(v)).reshape(-1)),axis=-1)

    ax.scatter(x,y,z,marker=marker)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.set_title(title)
    ax.autoscale_view()
    plt.show()
    im = get_img_from_fig(fig)
    plt.close('all')

    try:
        if(logger is not None):
            logger.addImage(im,title)
    except Exception as e:
        print(e)
    
    return im


def errhist(x,err,nbins=256,minv=-1,maxv=1,legend='f(x)'):
    """
    [Shows the histogram of a function (error) on an axis x]
    """

    bin_means, bin_edges, binnumber = scipy.stats.binned_statistic(x,err,statistic='mean', bins=nbins,range=[minv, maxv])
    bin_means[np.isnan(bin_means)] = -1
    bin_width = (bin_edges[1] - bin_edges[0])
    fig = plt.figure()
    plt.hlines(bin_means, bin_edges[:-1], bin_edges[1:], colors='g', lw=2,
            label=legend)
    # plt.plot((binnumber - 0.5) * bin_width, x_pdf, 'g.', alpha=0.5)
    plt.xlim(minv,maxv)
    ymin = bin_means.min()
    ymax = bin_means.max()
    margin = (ymax - ymin) * 0.1
    plt.ylim(ymin - margin, ymax + margin)
    plt.legend(fontsize=10)
    im = get_img_from_fig(fig)
    plt.close('all')
    return im


def plotOverlay(x,y1,y2,marker='.',xlabel='x',ylabel='y',legend=['Prediction','GT'],title='',xlim=None,ylim=None,step=None,logger=None,ptype='plot'):
    
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
    plt.close('all')

    try:
        if((not (logger is None)) and (not (step is None))):
            logger.addImage(im,title,step)
    except Exception as e:
        print(e)
    
    return im

def plotconfig(x=None,y=None,xlabel='x',ylabel='count',legend=['Prediction','GT'],title='',ax=None,fig=None,xlim=None,ylim=None,step=None,logger=None):
    # ax.set_xlabel(xlabel)
    # ax.set_ylabel(ylabel)
    # if(xlim =='margin' and x is not None):
    #     r = (x.max() + x.min()) / 2
    #     ax.set_xlim(x.min() - r * 0.3,x.max() + r * 0.3)
    # if(ylim is not None):
    #     ax.set_ylim(ylim[0],ylim[1])
    # ax.set_title(title)
    # ax.legend(legend)
    im = get_img_from_fig(fig)
    plt.close('all')
    try:
        if((not (logger is None)) and (not (step is None))):
            logger.addImage(im,title,step)
    except Exception as e:
        print(e)

    return im

def histogram(x,bins,**kwargs):
    fig, ax = plt.subplots()
    ax.hist(x,bins)
    return plotconfig(x=x,ax=ax,fig=fig,**kwargs)


def interpolationSeq(x,y,xs,ys):
    imgs = []
    for x0,y0 in zip(x,y):
        fig = plt.figure()
        plt.scatter(xs,ys,color='b')
        plt.scatter(x0,y0,color='r')
        plotim = get_img_from_fig(fig)
        plt.close('all')
        imgs.append(plotim)
    return np.stack(imgs,axis=0)
    
def saveInteractiveFig(fn,fig):
    with open(fn,'wb') as fd:
        pickle.dump(fig,fd)

def loadInteractiveFig(fn):
    with open(fn,"rb") as fd:
        return pickle.load(fd)
