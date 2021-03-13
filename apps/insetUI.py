from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QSlider, QHBoxLayout, QVBoxLayout,QGridLayout, QCheckBox
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import Qt
from cvgutils.ui import ImageWidget, RadioButtons, Slider
import cvgutils.Viz as viz
import os
import numpy as np
import h5py
import cv2
import argparse
import glob
import sys

class Parameters(QWidget):
    def __init__(self,parent,operateFunc,displayFunc,frameChangedFunc,opt):
        super().__init__(parent)
        self.operateFunc = operateFunc
        self.displayFunc = displayFunc
        self.frameChangedFunc = frameChangedFunc
        self.opt = opt

        self.initUI()

    def initUI(self):
        layout = QGridLayout(self)
        self.onlineChkbx = QCheckBox("Online")
        self.button = QPushButton("Render")
        self.button.clicked.connect(self.operateFunc)
        self.infoLabelCaptured = QLabel(self)
        self.infoLabelCaptured.setText('Info:')
        self.frameSlider = Slider(self.frameChangedFunc,self)
        self.frameSlider.slider.setRange(self.opt.start_id,self.opt.end_id)

        # Button | pixel patch image
        #        | point envmap
        #slider  | lineintegral, etc.
        layout.addWidget(self.button,0,0,3,1)
        layout.addWidget(self.infoLabelCaptured,1,2)
        layout.addWidget(self.frameSlider,3,1,1,2)
        layout.addWidget(self.onlineChkbx,4,0,1,1)
        
        self.renderRegion = ''
        self.setLayout(layout)


class AppInset(QWidget):

    def __init__(self,opt):
        super().__init__()
        self.title = 'Volume density viewer'
        self.left = 50
        self.top = 50
        self.width = 1500
        self.height = 600
        self.opt = opt
        
        
        self.initUI()

    def imageClicked(self, x, y):
        self.opt.renderCoord = '%i,%i,%i,%i' % (x,y,50,50)
    
    def frameChanged(self,b):

        self.imfn0 = self.opt.imfns0[b]
        self.imfn1 = self.opt.imfns1[b]
        self.imfn2 = self.opt.imfns2[b]
        self.imfn3 = self.opt.imfns3[b]
        self.imfn4 = self.opt.imfns4[b]

        self.im0 = cv2.imread(self.opt.imfns0[b],-1)
        self.im1 = cv2.imread(self.opt.imfns1[b],-1)
        self.im2 = cv2.imread(self.opt.imfns2[b],-1)
        self.im3 = cv2.imread(self.opt.imfns3[b],-1)
        self.im4 = cv2.imread(self.opt.imfns4[b],-1)

        im0 = self.im0
        im1 = self.im1
        im2 = self.im2
        im3 = self.im3
        im4 = self.im4

        if(im0 is not None):
            im0 = cv2.resize(im0,(im0.shape[1]//2,im0.shape[0]//2))
            self.label0.setImage(im0[:,:,::-1].copy())
        if(im1 is not None):
            im1 = cv2.resize(im1,(im1.shape[1]//2,im1.shape[0]//2))
            self.label1.setImage(im1[:,:,::-1].copy())
        if(im2 is not None):
            im2 = cv2.resize(im2,(im2.shape[1]//2,im2.shape[0]//2))
            self.label2.setImage(im2[:,:,::-1].copy())
        if(im3 is not None):
            im3 = cv2.resize(im3,(im3.shape[1]//2,im3.shape[0]//2))
            self.label3.setImage(im3[:,:,::-1].copy())
        if(im4 is not None):
            im4 = cv2.resize(im4,(im4.shape[1]//2,im4.shape[0]//2))
            self.label4.setImage(im4[:,:,::-1].copy())

    def imageMoved(self,x,y):
        pass
    def imageClicked(self,x,y):
        self.x0 = x*2
        self.y0 = y*2

    def store(self,a):
        def clipnstore(im,x0,x1,y0,y1,w,h,fn):
            im1 = im[y0:y1,x0:x1,:]
            im1 = cv2.resize(im1,(w,h))
            cv2.imwrite(fn,im1)


        if(self.x0 >0):
            w = 200
            h = 160
            for i in range(3,8):
                w2 = 10 * i
                h2 = 8 * i
                xlim = [self.x0-w2, self.x0+w2]
                ylim = [self.y0-h2, self.y0+h2]
                clipnstore(self.im0,xlim[0],xlim[1],ylim[0],ylim[1],w,h,os.path.join(self.opt.outdir,self.opt.methods[0]+'%03i_%03i.png' % (w2,h2)))
                clipnstore(self.im1,xlim[0],xlim[1],ylim[0],ylim[1],w,h,os.path.join(self.opt.outdir,self.opt.methods[1]+'%03i_%03i.png' % (w2,h2)))
                clipnstore(self.im2,xlim[0],xlim[1],ylim[0],ylim[1],w,h,os.path.join(self.opt.outdir,self.opt.methods[2]+'%03i_%03i.png' % (w2,h2)))
                clipnstore(self.im3,xlim[0],xlim[1],ylim[0],ylim[1],w,h,os.path.join(self.opt.outdir,self.opt.methods[3]+'%03i_%03i.png' % (w2,h2)))
                clipnstore(self.im4,xlim[0],xlim[1],ylim[0],ylim[1],w,h,os.path.join(self.opt.outdir,self.opt.methods[4]+'%03i_%03i.png' % (w2,h2)))

            cv2.imwrite(os.path.join(self.opt.outdir,self.opt.methods[0]+'.png'),self.im0)
            cv2.imwrite(os.path.join(self.opt.outdir,self.opt.methods[1]+'.png'),self.im1)
            cv2.imwrite(os.path.join(self.opt.outdir,self.opt.methods[2]+'.png'),self.im2)
            cv2.imwrite(os.path.join(self.opt.outdir,self.opt.methods[3]+'.png'),self.im3)
            cv2.imwrite(os.path.join(self.opt.outdir,self.opt.methods[4]+'.png'),self.im4)

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        layout = QGridLayout(self)
        
        self.x0 = -1
        self.y0 = -1
        
        im = np.zeros((10,10,3))
        self.label0 = ImageWidget(self,im,self.imageMoved,self.imageClicked)
        self.label1 = ImageWidget(self,im,self.imageMoved,self.imageClicked)
        self.label2 = ImageWidget(self,im,self.imageMoved,self.imageClicked)
        self.label3 = ImageWidget(self,im,self.imageMoved,self.imageClicked)
        self.label4 = ImageWidget(self,im,self.imageMoved,self.imageClicked)

        self.paramui = Parameters(self,self.store,self.frameChanged,self.frameChanged,self.opt)
        layout.addWidget(self.paramui,0,0,1,2)
        layout.addWidget(self.label0,1,0)
        layout.addWidget(self.label1,1,1)
        layout.addWidget(self.label2,1,2)
        layout.addWidget(self.label3,1,3)
        layout.addWidget(self.label4,1,4)
        self.setLayout(layout)
        self.show()
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Deploying command')
    #buddha2
    # parser.add_argument('--methods',type=str, default='buddha2_gt,buddha2_aug,buddha2_nrf,buddha2_pose', help='First set filename')
    # parser.add_argument('--imfns0',type=str, default='/home/mohammad/Projects/NRV/dataset/buddha2/testImgsExr-pointlight/*.png', help='First set filename')
    # parser.add_argument('--imfns1',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/buddha2/augmentation_method/render-video-latest-1/*.png', help='Second set filename')
    # parser.add_argument('--imfns2',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/buddha2/NRF_method/render-video-latest-1/*.png', help='Third set filename')
    # parser.add_argument('--imfns3',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/buddha2/pose_method/render-video-latest-1/*.png', help='Third set filename')

    # #buddha
    # parser.add_argument('--methods',type=str, default='buddha_gt,buddha_aug,buddha_no_aug,buddha_nrf,buddha_pose', help='First set filename')
    # parser.add_argument('--imfns0',type=str, default='/home/mohammad/Projects/NRV/dataset/buddha/testImgsExr-pointlight/*.png', help='First set filename')
    # parser.add_argument('--imfns1',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/buddha/augmentation_method/render-video-latest-1/fine_raycolor_pred_0*.png', help='Second set filename')
    # parser.add_argument('--imfns2',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/buddha/no_augmentation_method/render-video-latest-1/fine_raycolor_pred_0*.png', help='Second set filename')
    # parser.add_argument('--imfns3',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/buddha/NRF_method/render-video-latest-1/shadow_volume_*.png', help='Third set filename')
    # parser.add_argument('--imfns4',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/buddha/pose_method/render-video-latest-1/fine_raycolor_pred_*.png', help='Third set filename')

    # #girl
    # parser.add_argument('--methods',type=str, default='girl_aug,girl_aug,girl_no_aug,girl_nrf,girl_pose', help='First set filename')
    # parser.add_argument('--imfns0',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/girl/augmentation_method/render-video-latest-1/fine_raycolor_pred_0*.png', help='First set filename')
    # parser.add_argument('--imfns1',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/girl/augmentation_method/render-video-latest-1/fine_raycolor_pred_0*.png', help='Second set filename')
    # parser.add_argument('--imfns2',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/girl/no_augmentation_method/render-video-latest-1/fine_raycolor_pred_0*.png', help='Second set filename')
    # parser.add_argument('--imfns3',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/girl/NRF_method/render-video-latest-1/shadow_volume_*.png', help='Third set filename')
    # parser.add_argument('--imfns4',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/girl/pose_method/render-video-latest-1/fine_raycolor_pred_*.png', help='Third set filename')

    #pony
    parser.add_argument('--methods',type=str, default='pony_aug,pony_aug,pony_no_aug,pony_nrf,pony_pose', help='First set filename')
    parser.add_argument('--imfns0',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/pony/augmentation_method/render-video-latest-1/fine_raycolor_pred_0*.png', help='First set filename')
    parser.add_argument('--imfns1',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/pony/augmentation_method/render-video-latest-1/fine_raycolor_pred_0*.png', help='Second set filename')
    parser.add_argument('--imfns2',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/pony/no_augmentation_method/render-video-latest-1/fine_raycolor_pred_0*.png', help='Second set filename')
    parser.add_argument('--imfns3',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/pony/NRF_method/render-video-latest-1/shadow_volume_*.png', help='Third set filename')
    parser.add_argument('--imfns4',type=str, default='/home/mohammad/Projects/NRV/ICCVexpsCluster/pony/pose_method/render-video-latest-1/fine_raycolor_pred_*.png', help='Third set filename')

    parser.add_argument('--start_id',type=int, default=0, help='Third set filename')
    parser.add_argument('--end_id',type=int, default=500, help='Third set filename')

    parser.add_argument('--outdir',type=str, default='/home/mohammad/Projects/NRV/ICCVtex/Pointlight', help='Video filename filename')
    args = parser.parse_args()

    args.methods = args.methods.split(',')
    args.imfns0 = sorted(glob.glob(args.imfns0))
    args.imfns1 = sorted(glob.glob(args.imfns1))
    args.imfns2 = sorted(glob.glob(args.imfns2))
    args.imfns3 = sorted(glob.glob(args.imfns3))
    args.imfns4 = sorted(glob.glob(args.imfns4))

    from PyQt5.QtWidgets import QApplication
    from testScripts.ui import App
    app = QApplication(sys.argv)
    ex = AppInset(args)
    sys.exit(app.exec_())