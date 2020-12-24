import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QSlider
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, pyqtSignal
import PyQt5.QtCore as QtCore
import matplotlib.pyplot as plt
import cvgutils.Viz as viz
from cvgutils.ui import ImageWidget, RadioButtons
import numpy as np
import h5py
import os
import torch
import cv2

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 button - pythonspot.com'
        self.left = 50
        self.top = 50
        self.width = 1500
        self.height = 600
        self.initUI()

    def f1(self,e):
        return '%i %i' % (e.x(), e.y())

    def f2(self,e):
        return '%i %i' % (e.x(), e.y())

    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
        a = (np.random.rand(100,100,3) * 255).astype(np.uint8)
        # self.label = ImageWidget(self,a)
        self.rdbtn = RadioButtons(self.select, names=['blah1','blah2'], parent=self)
        self.show()

    def select(self):
        if(self.sender().isChecked()):
            print(self.sender().text())

    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')


if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
