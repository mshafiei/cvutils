from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QSlider, QHBoxLayout, QRadioButton, QVBoxLayout, QGridLayout, QFileDialog
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, pyqtSignal
import PyQt5.QtCore as QtCore
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class ImageViewer(QLabel):
    mouseMoved = pyqtSignal(int,int)
    mouseClicked = pyqtSignal(int,int)
    def __init__(self,parent,image,moveFunc=None,releaseFunc=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        assert len(image.shape) == 3
        self.setImage(image)
        if(moveFunc is not None):
            assert type(moveFunc) == list
            for i in moveFunc:
                self.mouseMoved.connect(i)
            
        if(releaseFunc is not None):
            assert type(releaseFunc) == list
            for i in releaseFunc:
                self.mouseClicked.connect(i)
                
    def setImage(self, im):
        h = im.shape[0]
        w = im.shape[1]
        qim = QImage(im, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap(qim))

    def mouseMoveEvent(self, e):
        self.mouseMoved.emit(e.x(),e.y())

    def mouseReleaseEvent(self, e):
        self.mouseClicked.emit(e.x(),e.y())

class ImageWidget(QWidget):

    def __init__(self,parent,image=None,moveFunc=None,releaseFunc = None):
        super().__init__(parent)
        layout = QGridLayout()
        self.saveButton = QPushButton("Save")
        self.saveButton.clicked.connect(self.saveImage)
        self.sl = QSlider(QtCore.Qt.Horizontal,self)
        self.sl.setMaximum(300)
        self.sl.setMinimum(25)
        self.sl.setValue(100)
        self.sl.setSingleStep(25)
        self.sl.valueChanged.connect(self.zoom)
        if(moveFunc is not None):
            moveFuncs = [self.f1, moveFunc]
        else:
            moveFuncs = [self.f1]
        if(releaseFunc is not None):
            releaseFuncs = [self.f2, releaseFunc]
        else:
            releaseFuncs = [self.f1]

        self.image = image
        h,w,_ = image.shape
        self.label = ImageViewer(self,image,moveFuncs,releaseFuncs)
        self.infoLabelHover = QLabel(self)
        self.infoLabelHover.setText('Hover x, %04i y, %04i' % (0,0))
        self.infoLabelCaptured = QLabel(self)
        self.infoLabelCaptured.setText('Captured x, %04i y, %04i' % (0,0))
        layout.addWidget(self.sl,0,0)
        layout.addWidget(self.infoLabelHover,0,1)
        layout.addWidget(self.infoLabelCaptured,0,2)
        layout.addWidget(self.saveButton,0,3)
        layout.addWidget(self.label,1,0,1,4)
        self.setLayout(layout)

    def saveImage(self, im):

        fn = QFileDialog.getSaveFileName(self, "Save image", './', ".exr")
        cv2.imwrite(fn[0] + fn[1],self.hdr[:,:,::-1].astype(np.float32))
        cv2.imwrite(fn[0] + fn[1].replace('exr','png'),self.image[:,:,::-1])

    def setImage(self, im):
        self.hdr = im
        self.image = (np.clip(im,0,1) ** (1/2.2) * 255).astype(np.uint8)
        # self.image = (np.clip(im,0,1) ** 4.2 * 255).astype(np.uint8)
        self.label.setImage(self.image)

    def f1(self,x,y):
        x = x / (self.sl.value() / 100.0)
        y = y / (self.sl.value() / 100.0)
        txt = 'Hover x, %04i y, %04i' % (x, y)
        self.infoLabelHover.setText(txt)
        
    def f2(self,x,y):
        x = x / (self.sl.value() / 100.0)
        y = y / (self.sl.value() / 100.0)
        txt = 'Cpatured x, %04i y, %04i' % (x, y)
        self.infoLabelCaptured.setText(txt)

    def zoom(self):
        im = self.image.copy()
        dsize = (int(im.shape[1]*self.sl.value() / 100.0),int(im.shape[0]*self.sl.value() / 100.0))
        im = cv2.resize(im,dsize)
        h = im.shape[0]
        w = im.shape[1]
        self.setImage(im)

class RadioButtons(QWidget):

   def __init__(self, selectedFunc, names=['Button1', 'Button2'], parent=None):
        super().__init__(parent)
        """[Creates a layout containing a set of radio buttons]

        :param selectedFunc: [Callback function ex: Func(self): self.sender().isChecked()]
        :type selectedFunc: [function]
        :param names: [Text of Buttons], defaults to ['Button1', 'Button2']
        :type names: list, optional
        :param parent: [parent widget], defaults to None
        :type parent: [QWidget], optional
        """
            
        layout = QHBoxLayout()
        self.buttons = dict()
        for name in names:
            self.buttons[name] = QRadioButton(name)
            self.buttons[name].toggled.connect(selectedFunc)
            layout.addWidget(self.buttons[name])
        self.buttons[name].setChecked(True)
        self.setLayout(layout)

class Slider(QWidget):
    def __init__(self, valueChangeFunc, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout()
        self.slider = QSlider(Qt.Horizontal)
        self.label = QLabel('')
        layout.addWidget(self.slider)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.slider.valueChanged.connect(valueChangeFunc)
        self.slider.valueChanged.connect(self.valueChange)

    def valueChange(self,b):
        self.label.setText(str(b))