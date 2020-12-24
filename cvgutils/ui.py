from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from PyQt5.QtGui import QIcon, QImage, QPixmap
from PyQt5.QtCore import pyqtSlot, pyqtSignal

class ImageViewer(QLabel):
    mouseMoved = pyqtSignal()
    def __init__(self,parent):
        super().__init__(parent)
        self.setMouseTracking(True)

    def mouseMoveEvent(self, e):
        self.mouseMoved.emit()
        QLabel.mousePressEvent(self, e)
