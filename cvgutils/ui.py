from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QSlider, QHBoxLayout, QRadioButton, QVBoxLayout, QGridLayout
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, pyqtSignal
import PyQt5.QtCore as QtCore
import cv2
import numpy as np

class ImageViewer(QLabel):
    mouseMoved = pyqtSignal(int,int)
    mouseClicked = pyqtSignal(int,int)
    def __init__(self,parent,image,moveFunc=None,releaseFunc=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        assert len(image.shape) == 3
        h,w,_ = image.shape
        qim = QImage(image, w, h, 3 * w, QImage.Format_RGB888)
        self.setPixmap(QPixmap(qim))
        if(moveFunc is not None):
            self.mouseMoved.connect(moveFunc)
            
        if(moveFunc is not None):
            self.mouseClicked.connect(releaseFunc)

    def mouseMoveEvent(self, e):
        self.mouseMoved.emit(e.x(),e.y())

    def mouseReleaseEvent(self, e):
        self.mouseClicked.emit(e.x(),e.y())

class ImageWidget(QWidget):

    def __init__(self,parent,image):
        super().__init__(parent)
        layout = QGridLayout()
        self.sl = QSlider(QtCore.Qt.Horizontal,self)
        self.sl.setMaximum(300)
        self.sl.setMinimum(25)
        self.sl.setValue(100)
        self.sl.setSingleStep(25)
        self.sl.valueChanged.connect(self.zoom)
        self.sl.setGeometry(0,0,200,25)
        self.image = image
        h,w,_ = image.shape
        self.label = ImageViewer(self,image,self.f1,self.f2)
        self.label.setGeometry(0,25,w,h)
        self.infoLabel = QLabel(self)
        self.infoLabel.setGeometry(200,0,200,20)
        self.setGeometry(0,0,600,600)
        layout.addWidget(self.sl,0,0)
        layout.addWidget(self.infoLabel,0,1)
        layout.addWidget(self.label,1,0,1,2)
        self.setLayout(layout)



    def f1(self,x,y):
        x = x / self.sl.value() / 100.0
        y = y / self.sl.value() / 100.0
        txt = 'x, %04i y, %04i' % (x, y)
        self.infoLabel.setText(txt)
        
    def f2(self,x,y):
        x = x / self.sl.value() / 100.0
        y = y / self.sl.value() / 100.0
        txt = 'x, %04i y, %04i' % (x, y)
        self.infoLabel.setText(txt)

    def zoom(self):
        im = self.image.copy()
        dsize = (int(im.shape[1]*self.sl.value() / 100.0),int(im.shape[0]*self.sl.value() / 100.0))
        im = cv2.resize(im,dsize)
        h = im.shape[0]
        w = im.shape[1]
        qim = QImage(im, w, h, 3 * w, QImage.Format_RGB888)
        self.label.setPixmap(QPixmap(qim))
        self.label.setGeometry(self.label.x(),self.label.y(),w,h)

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
