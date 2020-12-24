# #create one label
# #update it with images
# #create callback for mouse hover and click
# #register the clicked point
# #use a button for operations on registered pixel coord
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QSlider
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, pyqtSignal
import PyQt5.QtCore as QtCore
import matplotlib.pyplot as plt
import cvgutils.Viz as viz
import numpy as np
import h5py
import os
import torch
import cv2
# import cvgutils.Viz as viz
# from cvgutils import ui

class ImageViewer(QLabel):
    mouseMoved = pyqtSignal(int,int)
    mouseClicked = pyqtSignal(int,int)
    def __init__(self,parent):
        super().__init__(parent)
        self.setMouseTracking(True)
        

    def mouseMoveEvent(self, e):
        self.mouseMoved.emit(e.x(),e.y())

    def mouseReleaseEvent(self, e):
        self.mouseClicked.emit(e.x(),e.y())

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 button - pythonspot.com'
        self.left = 50
        self.top = 50
        self.width = 1500
        self.height = 600
        self.initUI()
    
    def print(self,x,y):
        x = int(x / self.sl.value())
        y = int(y / self.sl.value())
        text = f'x: {x},  y: {y}'
        print(text)

    def print2(self,x,y):
        #plot opacity, transmittance
        #plot siren opacity, transmittance
        x = int(x / self.sl.value())
        y = int(y / self.sl.value())
        transmittance_siren = self.f['transmittance_pred'][self.imid][y,x,:,0]
        transmittance_GT = self.f['fine_acc_alpha'][self.imid][y,x,:,0]
        if('weight_pred' in self.f):
            weight_pred = self.f['weight_pred'][self.imid][y,x,:,0]
            weight_gt = self.f['weight_GT'][self.imid][y,x,:,0]
        if('fine_alpha' in self.f):
            fine_alpha = self.f['fine_alpha'][self.imid][y,x,:,0]
        
        fine_samples_t = self.f['fine_samples_t'][self.imid][y,x,:,0]
        if('fine_samples_pred' in self.f):
            fine_raycolor_pred = self.f['fine_samples_pred'][self.imid][y,x,:,0]
        if('fine_samples_GT' in self.f):
            fine_raycolor_gt = self.f['fine_samples_GT'][self.imid][y,x,:,0]
        if('fine_samples_color' in self.f):
            fine_samples_color = self.f['fine_samples_color'][self.imid][y,x,:,0]
        
        # fine_rayposdir = self.f['fine_rayposdir'][self.imid][y,x,:,:]
        

        #log transmittance
        


        eps = 1e-5
        logrange = np.log(1+eps) - np.log(eps)
        im = viz.plotOverlay(fine_samples_t,-np.log(transmittance_siren+1e-5) / logrange,-np.log(transmittance_GT+1e-5) / logrange,xlabel='t',ylabel='log transmittance',title='log Transmittance')
        if(not os.path.exists('renderout')):
            os.makedirs('renderout')
        cv2.imwrite('renderout/log_transmittance.png',im)

        #transmittance
        if('transmittance_siren' in locals() and 'transmittance_GT' in locals()):
            im = viz.plotOverlay(fine_samples_t,transmittance_siren,transmittance_GT,xlabel='t',title='Transmittance ReLU',ylabel='Transmittance')
            if(not os.path.exists('renderout')):
                os.makedirs('renderout')
            cv2.imwrite('renderout/transmittance.png',im)

        if('weight_pred' in locals() and 'weight_gt' in locals()):
            im = viz.plotOverlay(fine_samples_t,weight_pred,weight_gt,xlabel='t',ylabel='Weight',title='Weight ReLU')
            if(not os.path.exists('renderout')):
                os.makedirs('renderout')
            cv2.imwrite('renderout/Weight.png',im)

        if('fine_alpha' in locals()):
            im = viz.plot(fine_samples_t,fine_alpha,'.','t','alpha','alpha GT')
            if(not os.path.exists('renderout')):
                os.makedirs('renderout')
            cv2.imwrite('renderout/alpha.png',im)

        if('fine_samples_color' in locals()):
            im = viz.plot(fine_samples_t,fine_samples_color,'.','t','sampled color','sampled color gt')
            if(not os.path.exists('renderout')):
                os.makedirs('renderout')
            cv2.imwrite('renderout/sampled_color_gt.png',im)

        if('fine_samples_color' in locals() and 'weight_gt' in locals()):
            # color_gt = fine_samples_color * weight_gt
            # color_pred = fine_samples_color * weight_pred
            color_gt = weight_gt
            color_pred = weight_pred
            im = viz.plotOverlay(fine_samples_t,color_pred,color_gt,xlabel='t',ylabel='Intensity',title='color')
            if(not os.path.exists('renderout')):
                os.makedirs('renderout')
            cv2.imwrite('renderout/color.png',im)

        
    def zoom(self):
        im = self.imorig.copy()
        dsize = (int(im.shape[1]*self.sl.value()),int(im.shape[0]*self.sl.value()))
        im = cv2.resize(im,dsize)
        im = np.clip(im,0,1) ** (1/2.2) * 255
        im = im.astype(np.uint8)[:,:,:].copy()
        h = im.shape[0]
        w = im.shape[1]
        qim = QImage(im, w, h, 3 * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap(qim))
        self.label.setGeometry(QtCore.QRect(0,0,w,h))

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)

        self.label = ImageViewer(self)
        self.imid = 0
        self.f = h5py.File('./hdf5/hdf5-w99-497.hdf5','r')
        self.imorig = self.f['fine_raycolor_pred'][self.imid].copy()
        im = np.clip(self.imorig,0,1) ** (1/2.2) * 255
        im = im.astype(np.uint8)[:,:,:].copy()
        cv2.imwrite('renderout/im.png',im[:,:,::-1])

        im2 = self.f['fine_raycolor'][self.imid].copy()
        im2 = np.clip(im2,0,1) ** (1/2.2) * 255
        im2 = im2.astype(np.uint8)[:,:,:].copy()
        cv2.imwrite('renderout/imGT.png',im2[:,:,::-1])
        
        transmittance_siren = self.f['transmittance_pred'][self.imid]
        transmittance_GT = self.f['fine_acc_alpha'][self.imid]

        if('weight_pred' in self.f):
            weight_pred = self.f['weight_pred'][self.imid]
            weight_gt = self.f['weight_GT'][self.imid]
        
            # fine_samples_t = self.f['fine_samples_t'][self.imid]
            # fine_rayposdir = self.f['fine_rayposdir'][self.imid]
            wpim = weight_pred.sum(-2)
            wgim = weight_gt.sum(-2)
            werr = (((weight_pred - weight_gt) ** 2).sum(-2) ** 0.5)

            cv2.imwrite('renderout/wpim.exr',wpim.astype(np.float32))
            cv2.imwrite('renderout/wgim.exr',wgim.astype(np.float32))
            cv2.imwrite('renderout/werr.exr',werr.astype(np.float32))

        self.h = im.shape[0]
        self.w = im.shape[1]

        qim = QImage(im, self.w, self.h, 3 * self.w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap(qim))
        # self.setCentralWidget(self.label)
        self.label.move(0,0)

        
        # fine_acc_alpha_pred = self.f['fine_acc_alpha_pred'][self.imid].sum(1).reshape(self.h,self.w,1).repeat(3,2)
        # fine_acc_alpha = self.f['fine_acc_alpha'][self.imid].sum(1).reshape(self.h,self.w,1).repeat(3,2)
        # fine_alpha = self.f['fine_alpha'][self.imid].sum(1).reshape(self.h,self.w,1).repeat(3,2)
        # fine_alpha_pred = self.f['fine_alpha_pred'][self.imid].sum(1).reshape(self.h,self.w,1).repeat(3,2)
        # sidebyside = np.concatenate((fine_acc_alpha_pred,fine_acc_alpha,fine_alpha_pred,fine_alpha),axis=1)
        # sidebyside = np.clip(sidebyside*0.001,0,1) ** (1/2.2) * 255 
        # sidebyside = sidebyside.astype(np.uint8)
        # label2 = ImageViewer(self)
        # qim2 = QImage(sidebyside, sidebyside.shape[1], sidebyside.shape[0], 3 * sidebyside.shape[1], QImage.Format_RGB888)
        # label2.setPixmap(QPixmap(qim2))
        # label2.move(0,self.h)

        self.sl = QSlider(QtCore.Qt.Horizontal,self)
        self.sl.setMaximum(3)
        self.sl.setMinimum(1)
        self.sl.setSingleStep(0.2)
        self.sl.valueChanged.connect(self.zoom)
        self.sl.move(1024,0)

        button = QPushButton('PyQt5 button', self)
        button.setToolTip('This is an example button')
        button.move(1024,70)
        button.clicked.connect(self.on_click)
        self.label.mouseMoved.connect(self.print)
        self.label.mouseClicked.connect(self.print2)
        
        self.show()

    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')

if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
