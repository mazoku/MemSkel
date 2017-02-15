
from PyQt4 import QtGui, QtCore
import numpy as np
import cv2


class MyCanvas(QtGui.QLabel):#QWidget):
    def __init__(self, parent=None):
        super(MyCanvas, self).__init__(parent)
        self.marking = False  # flag whether the user is marking seed points
        self.myPenWidth = 10
        self.myPenColor = QtCore.Qt.red
        self.image = QtGui.QImage()
        self.lastPoint = QtCore.QPoint()
        self.mask = QtGui.QImage()
        self.canvas_size = (600, 600)

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            print event.pos()
            self.lastPoint = event.pos()
            self.marking = True

    def mouseMoveEvent(self, event):
        if (event.buttons() & QtCore.Qt.LeftButton) and self.marking:
            self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self.marking:
            self.drawLineTo(event.pos())
            self.marking = False

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        # painter.begin(self)
        painter.drawImage(event.rect(), self.image)
        # self.set_img_vis(self.image)
        # painter.end()

    def drawLineTo(self, endPoint):
        painter = QtGui.QPainter(self.image)
        painter.setPen(QtGui.QPen(self.myPenColor, self.myPenWidth, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap,
                                  QtCore.Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
        self.modified = True

        rad = self.myPenWidth / 2 + 10
        self.update(QtCore.QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
        self.lastPoint = QtCore.QPoint(endPoint)
        self.update()

    # def resizeEvent(self, event):
    #     if self.width() > self.image.width() or self.height() > self.image.height():
    #         newWidth = max(self.width() + 128, self.image.width())
    #         newHeight = max(self.height() + 128, self.image.height())
    #         self.resizeImage(self.image, QtCore.QSize(newWidth, newHeight))
    #         self.update()
    #
        # super(MyCanvas, self).resizeEvent(event)

    # def resizeImage(self, image, newSize):
    #     if image.size() == newSize:
    #         return
    # #
    #     newImage = QtGui.QImage(newSize, QtGui.QImage.Format_RGB32)
    #     newImage.fill(QtGui.qRgb(255, 255, 255))
    #     painter = QtGui.QPainter(newImage)
    #     painter.drawImage(QtCore.QPoint(0, 0), image)
    #     self.image = newImage

    # def set_img_vis(self, img):
    #     self.img_vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    #     qimg = QtGui.QImage(self.img_vis.data, self.img_vis.shape[1], self.img_vis.shape[0], QtGui.QImage.Format_RGB888)
    #     pixmap = QtGui.QPixmap.fromImage(qimg)
    #     pixmap = pixmap.scaled(QtCore.QSize(*self.canvas_size), QtCore.Qt.KeepAspectRatio)
    #     # pixmap = pixmap.scaled(self.canvas_L.size(), Qt.KeepAspectRatio)
    #     self.setPixmap(pixmap)


if __name__ == '__main__':
    class MainWindow(QtGui.QMainWindow):
        def __init__(self):
            super(MainWindow, self).__init__()
            # self.canvas = MyCanvas()
            # self.setCentralWidget(self.canvas)
            img = cv2.imread('data/icons/kky.png')
            # # self.resize(img.shape[1], img.shape[0])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #
            qimg = QtGui.QImage(img.data, img.shape[1], img.shape[0], QtGui.QImage.Format_RGB888)
            # painter = QtGui.QPainter(qimg)
            # painter.drawImage(QtCore.QPoint(0, 0), qimg)
            self.canvas.image = qimg
            # print 'canvas:', self.canvas.size()
            # print 'window:', self.size()
            # self.canvas.image = QtCore.QPixmap('data/icons/kky.png')

            # self.canvas.setPixmap(QtGui.QPixmap('data/icons/kky.png'))
            self.image = QtGui.QLabel()
            self.image.setPixmap(QtGui.QPixmap('data/icons/kky.png'))
            self.image.mousePressEvent = self.getPos
            self.setCentralWidget(self.image)

        def getPos(self, event):
            print event.pos()

    import sys

    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())