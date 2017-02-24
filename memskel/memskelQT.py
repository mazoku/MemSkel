from __future__ import division
__author__ = 'Ryba'

# TODO: zpravy do statusbaru
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
from MyImageViewer import ImageViewerQt
from segmentator import Segmentator

# ----
from constants import *
# OBJ_COLOR = [255, 0, 0]  # QtCore.Qt.red
# BGD_COLOR = [0, 0, 255]  # QtCore.Qt.blue
# OBJ_SEED_LBL = 1
# BGD_SEED_LBL = 2


class MainWindow(QtGui.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        # self.canvas_size = (600, 600)
        # self.slice_slider_F.hide()

        self.actual_idx = 0  # index of actual (displayed) slice/frame of data
        self.data_fname = None
        self.data = ImageData()
        self.segmentator = Segmentator()
        self.segmentator.data = self.data
        self.marking = False  # flag whether the user is marking seed points
        self.pen_color = None
        self.seed_lbl = None
        # self.x

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
        # self.image = cv2.cvtColor(cv2.imread(logo_fname), cv2.COLOR_BGR2RGB)
        # self.seeds = np.zeros(self.image.shape[:2], dtype=np.uint8)
        # self.img_vis = QtGui.QPixmap(logo_fname)
        # self.canvas_L.setPixmap(self.img_vis)
        self.canvas_GV = ImageViewerQt()
        self.canvas_GV.setImage(QtGui.QPixmap(logo_fname))
        self.frame_7.layout().insertWidget(0, self.canvas_GV)

    # def resize_canvas_event(self, params):
    #     # self.set_img_vis(self.data.data[self.actual_idx, ...])
    #     self.canvas_size = self.canvas_L.size()
    #     self.center()

    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def wheelEvent(self, event):
        x = int(event.delta() / 120)
        if not self.marking:
            if x > 0:
                idx = min(self.actual_idx + 1, self.data.n_slices - 1)
                # self.actual_idx = min(self.actual_idx + 1, self.data.n_slices - 1)
            elif x< 0:
                idx = max(0, self.actual_idx - 1)
                # self.actual_idx = max(0, self.actual_idx - 1)
            # self.set_img_vis(self.data.image[self.actual_idx, ...])
            self.slice_SB.setValue(idx)
        else:
            self.linewidth_SB.setValue(self.linewidth_SB.value() + x)

    def handleLeftClick(self, x, y):
        self.last_pt = (int(x), int(y))
        # print 'clicked: {}'.format((x, y))

    def handleMouseMove(self, x, y):
        pt = (int(x), int(y))
        # print 'moved to: {}'.format(pt)
        if self.marking:
            linewidth = self.linewidth_SB.value()
            # cv2.line(self.data.seeds[self.actual_idx, ...], self.last_pt, pt, 255, linewidth)
            cv2.line(self.data.seeds[self.actual_idx, ...], self.last_pt, pt, self.seed_lbl, linewidth)
            self.last_pt = pt
            # self.image_vis = cv2.cvtColor(self.data.image[self.actual_idx, ...], cv2.COLOR_GRAY2RGB)
            # self.image_vis[np.nonzero(self.data.seeds[self.actual_idx, ...])] = self.pen_color#[255, 0, 0]
            # self.qimage = QtGui.QImage(self.image_vis.data, self.data.n_cols, self.data.n_rows, QtGui.QImage.Format_RGB888)
            self.create_img_vis()
            self.canvas_GV.setImage(self.qimage)

    def set_marking(self, value):
        self.marking = value
        if self.marking:
            self.canvas_GV.setCursor(QtCore.Qt.CrossCursor)
        else:
            self.canvas_GV.setCursor(QtCore.Qt.ArrowCursor)

    def mark_seeds(self, type):
        if type == 'o':
            btn = self.segment_obj_BTN
            self.segment_bgd_BTN.setChecked(False)
            self.pen_color = OBJ_COLOR
            self.seed_lbl = OBJ_SEED_LBL
            # self.marking = True
            self.set_marking(True)
            self.canvas_GV.leftMouseButtonPressed.connect(self.handleLeftClick)
            self.canvas_GV.mouseMoved.connect(self.handleMouseMove)
        elif type == 'b':
            btn = self.segment_bgd_BTN
            self.segment_obj_BTN.setChecked(False)
            self.pen_color = BGD_COLOR
            self.seed_lbl = BGD_SEED_LBL
            # self.marking = True
            self.set_marking(True)
            self.canvas_GV.leftMouseButtonPressed.connect(self.handleLeftClick)
            self.canvas_GV.mouseMoved.connect(self.handleMouseMove)

        if not btn.isChecked():
            # self.marking = False
            self.set_marking(False)
            self.canvas_GV.leftMouseButtonPressed.disconnect()
            self.canvas_GV.mouseMoved.disconnect()
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
        self.create_img_vis()
        self.canvas_GV.setImage(self.qimage)
        # self.set_img_vis(self.image_vis)

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
        self.set_img_vis(self.data.image[self.actual_idx, ...])

    def create_img_vis(self):
        self.image_vis = cv2.cvtColor(self.data.image[self.actual_idx, ...], cv2.COLOR_GRAY2RGB)
        # self.image_vis[np.nonzero(self.data.seeds[self.actual_idx, ...])] = [255, 0, 0]
        self.image_vis[np.nonzero(self.data.seeds[self.actual_idx, ...] == OBJ_SEED_LBL)] = OBJ_COLOR
        self.image_vis[np.nonzero(self.data.seeds[self.actual_idx, ...] == BGD_SEED_LBL)] = BGD_COLOR
        self.qimage = QtGui.QImage(self.image_vis.data, self.image_vis.shape[1], self.image_vis.shape[0], QtGui.QImage.Format_RGB888)
        # self.canvas_GV.setImage(self.qimage)

    def set_img_vis(self, img):
        self.img_vis = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        # qimg = QtGui.QImage(self.img_vis.data, self.img_vis.shape[1], self.img_vis.shape[0], QtGui.QImage.Format_RGB888)
        # pixmap = QtGui.QPixmap.fromImage(qimg)
        # pixmap = pixmap.scaled(QtCore.QSize(*self.canvas_size), QtCore.Qt.KeepAspectRatio)
        # # pixmap = pixmap.scaled(self.canvas_L.size(), Qt.KeepAspectRatio)
        # self.canvas_L.setPixmap(pixmap)
        self.qimage = QtGui.QImage(self.img_vis.data, self.img_vis.shape[1], self.img_vis.shape[0], QtGui.QImage.Format_RGB888)
        self.canvas_GV.setImage(self.qimage)


if __name__ == '__main__':
    app = QtGui.QApplication(sys.argv)
    memskel = MainWindow()
    memskel.load_data(fname='data/smallStack.tif')
    memskel.show()
    app.exec_()