import sys
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel, QSlider
from PyQt5.QtGui import QIcon, QImage, QPixmap
import PyQt5.QtCore as QtCore
from cvgutils.ui import ImageWidget
import numpy as np
import cv2

class App(QWidget):

    def __init__(self):
        super().__init__()
        self.title = 'Volume density viewer'
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
        self.label = ImageWidget(self,a)

        self.show()



if __name__ == '__main__':

    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())
