from __future__ import division
__author__ = 'Ryba'

# # py2exe and libtiff stuff
# import sys
# # sys.path += ['.']
# # from ctypes import util
# #---------------------------
#
# import Tkinter as tk
# # import ttk
# import os
# import tkFileDialog
# import numpy as np
#
# # import matplotlib.pyplot as plt
# # import pymorph as pm
#
# # from pylab import *
# import matplotlib.pyplot as plt
# from pylab import Circle, Axes, figaspect
#
# from PIL import Image, ImageTk #, PngImagePlugin, TiffImagePlugin
# # Image._initialized = 2  # otherwise PIL cannot identify image file after py2exe debuging
# from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
#
# # from scipy.sparse.csgraph import _validation #otherwise could not be compiled by py2exe
# from scipy import interpolate
#
# import skimage.exposure as skiexp
# import skimage.morphology as skimor
#
# import MySketcher
# import memskel_tools as mt
# from libtiff import TIFFimage #, TIFF
# import ProgressMeter as pmeter
#
# import logging

import sys
# from PyQt4.QtGui import QMainWindow, QApplication, QImage, QPixmap, QFileDialog, QPainter, QPen
# from PyQt4.QtCore import Qt, QSize
from PyQt4 import QtCore, QtGui
from memskel_GUI_QT import Ui_MainWindow

import numpy as np
import skimage.io as skiio
import cv2

from imagedata import ImageData

# ----
OBJ_COLOR = QtCore.Qt.red
BGD_COLOR = QtCore.Qt.blue


class MainWindow(QtGui.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        self.canvas_size = (600, 600)
        # self.slice_slider_F.hide()

        self.actual_idx = 0  # index of actual (displayed) slice/frame of data
        self.data_fname = None
        self.data = ImageData()
        self.marking = False  # flag whether the user is marking seed points

        # OVERRIDING ----
        # self.canvas_L.resized.connect(self.resize_canvas_event)
        # self.canvas_L.resizeEvent = self.resize_canvas_event

        # SIGNALS ----
        self.open_BTN.clicked.connect(self.load_data)
        self.slice_SB.valueChanged.connect(self.slice_SB_changed)
        self.segment_obj_BTN.clicked.connect(lambda: self.mark_seeds('o'))
        self.segment_bgd_BTN.clicked.connect(lambda: self.mark_seeds('b'))

        # display default image
        logo_fname = 'data/icons/kky.png'
        self.img_vis = QtGui.QPixmap(logo_fname)
        self.canvas_L.setPixmap(self.img_vis)

    # def resize_canvas_event(self, params):
    #     # self.set_img_vis(self.data.data[self.actual_idx, ...])
    #     self.canvas_size = self.canvas_L.size()

    def mark_seeds(self, type):
        if type == 'o':
            btn = self.segment_obj_BTN
            self.segment_bgd_BTN.setChecked(False)
            color = OBJ_COLOR
        elif type == 'b':
            btn = self.segment_bgd_BTN
            self.segment_obj_BTN.setChecked(False)
            color = BGD_COLOR

        if not btn.isChecked():
            return

    # def mouse_press_event(self, event):
    #     if event.button() == QtCore.Qt.LeftButton:
    #         self.lastPoint = event.pos()
    #         self.marking = True
    #
    # def mouse_move_event(self, event):
    #     if (event.buttons() & QtCore.Qt.LeftButton) and self.marking:
    #         self.drawLineTo(event.pos())
    #
    # def mouse_release_event(self, event):
    #     if event.button() == QtCore.Qt.LeftButton and self.marking:
    #         self.drawLineTo(event.pos())
    #         self.marking = False
    #
    # def drawLineTo(self, endPoint):
    #     painter = QtGui.QPainter(self.image)
    #     painter.setPen(QtCore.QPen(self.myPenColor, self.myPenWidth, QtCore.Qt.SolidLine,
    #                                QtCore.Qt.RoundCap, QtCore.Qt.RoundJoin))
    #     painter.drawLine(self.lastPoint, endPoint)
    #     self.modified = True
    #
    #     rad = self.myPenWidth / 2 + 2
    #     self.update(QtCore.QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad, +rad, +rad))
    #     self.lastPoint = QtCore.QPoint(endPoint)

    def slice_SB_changed(self, value):
        self.actual_slice_LBL.setText(str(value + 1))
        self.actual_idx = value
        self.set_img_vis(self.data.data[self.actual_idx, ...])

    def load_data(self, fname=None):
        if fname is None:
            self.data_fname = str(QtGui.QFileDialog.getOpenFileName(self, 'Open file', 'data/', "Image files (*.jpg *.gif *.tiff *.tif)"))
        else:
            self.data_fname = fname
        self.data.load(self.data_fname)

        # scrolbar update
        self.slice_SB.setMaximum(self.data.n_slices - 1)
        # self.min_slice_idx_LBL.setText(str(1))
        self.max_slice_LBL.setText(str(self.data.n_slices))
        self.slice_SB_changed(0)
        self.set_img_vis(self.data.data[self.actual_idx, ...])

    def set_img_vis(self, img):
        self.img_vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        qimg = QtGui.QImage(self.img_vis.data, self.img_vis.shape[1], self.img_vis.shape[0], QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        pixmap = pixmap.scaled(QtCore.QSize(*self.canvas_size), QtCore.Qt.KeepAspectRatio)
        # pixmap = pixmap.scaled(self.canvas_L.size(), Qt.KeepAspectRatio)
        self.canvas_L.setPixmap(pixmap)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    memskel = MainWindow()
    memskel.load_data(fname='data/smallStack.tif')
    memskel.show()
    app.exec_()