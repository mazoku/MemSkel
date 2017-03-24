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
import time
# from PyQt4.QtGui import QMainWindow, QApplication, QImage, QPixmap, QFileDialog, QPainter, QPen
# from PyQt4.QtCore import Qt, QSize
from PyQt4 import QtCore, QtGui
from memskel_GUI_QT import Ui_MainWindow

from collections import namedtuple
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
        # self.circ_roi = [0, 0, 0]  # circular roi
        self.circ_roi = {'center': (0, 0), 'radius': self.circ_roi_radius_SB.value()}
        self.rect_roi = {'pt1': None, 'pt2': None}
        self.eraser_roi = {'center': (0, 0), 'radius': self.eraser_roi_radius_SB.value()}
        self.rect_roi_pts_clicked = 0
        self.disp_membrane = self.disp_membrane_BTN.isChecked()
        self.disp_seeds = self.disp_seeds_BTN.isChecked()
        # self.x

        # OVERRIDING ----
        # self.canvas_L.resized.connect(self.resize_canvas_event)
        # self.canvas_L.resizeEvent = self.resize_canvas_event

        # SIGNALS ----
        self.open_BTN.clicked.connect(self.load_data)
        self.slice_SB.valueChanged.connect(self.slice_SB_changed)
        self.mark_obj_BTN.clicked.connect(lambda: self.mark_seeds('o'))
        self.mark_bgd_BTN.clicked.connect(lambda: self.mark_seeds('b'))
        self.segment_img_BTN.clicked.connect(lambda: self.segment_img(twoD=True))

        self.disp_membrane_BTN.clicked.connect(self.disp_membrane_clicked)
        self.disp_seeds_BTN.clicked.connect(self.disp_seeds_clicked)
        self.disp_roi_BTN.clicked.connect(self.disp_roi_clicked)

        self.circle_ROI_BTN.clicked.connect(self.define_circle_roi)
        self.circ_roi_radius_SB.valueChanged.connect(self.circ_radius_SB_changed)
        self.rectangle_ROI_BTN.clicked.connect(self.define_rect_roi)

        self.threshold_BTN.clicked.connect(self.threshold_clicked)
        self.threshold_SB.valueChanged.connect(self.threshold_changed)

        self.eraser_BTN.clicked.connect(self.eraser_clicked)
        self.eraser_roi_radius_SB.valueChanged.connect(self.eraser_radius_SB_changed)

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

    def eraser_clicked(self):
        if self.eraser_BTN.isChecked():
            self.canvas_GV.setMouseTracking(True)
            self.canvas_GV.leftMouseButtonPressed.connect(self.eraser_left_click)
            self.canvas_GV.mouseMoved.connect(self.eraser_mouse_move)
        else:
            self.canvas_GV.setMouseTracking(False)
            self.canvas_GV.leftMouseButtonPressed.disconnect()
            self.canvas_GV.mouseMoved.disconnect()
            self.create_img_vis(update=True)

    def eraser_radius_SB_changed(self, x):
        self.eraser_roi['radius'] = x
        self.create_img_vis(update=True)

    def eraser_mouse_move(self, x, y, btn):
        x = int(round(x))
        y = int(round(y))
        self.eraser_roi['center'] = (x, y)

        # delete seeds and segmentation inside ROI if left mouse is pressed
        if btn == QtCore.Qt.LeftButton:
            self.erase()
        self.create_img_vis(update=True)

    def eraser_left_click(self, x, y):
        self.erase()
        self.create_img_vis(update=True)

    def erase(self):
        roi = np.ones((self.data.n_rows, self.data.n_cols), dtype=np.uint8)
        cv2.circle(roi, self.eraser_roi['center'], self.eraser_roi['radius'], 0, -1)
        self.data.segmentation[self.actual_idx, ...] *= roi
        self.data.seeds[self.actual_idx, ...] *= roi

    def update_segmentation(self, mask):
        self.data.segmentation[self.actual_idx, ...] = mask
        self.create_img_vis(update=True)
        self.canvas_GV.repaint()

    def threshold_clicked(self):
        if self.threshold_BTN.isChecked():
            self.disp_roi_BTN.setChecked(True)
            t = self.segmentator.get_threshold(self.data.image[self.actual_idx, ...], pt=70)
            self.threshold_SB.setValue(t)
            self.data.update_roi()
            self.create_img_vis(update=True)

    def threshold_changed(self, x):
        self.data.thresh_roi[self.actual_idx, ...] = self.data.image[self.actual_idx, ...] > x
        self.data.update_roi()
        self.create_img_vis(update=True)

    def disp_roi_clicked(self):
        self.create_img_vis(update=True)

    def circ_radius_SB_changed(self, x):
        self.circ_roi['radius'] = x
        self.create_img_vis(update=True)

    def define_circle_roi(self):
        if self.circle_ROI_BTN.isChecked():
            self.canvas_GV.setMouseTracking(True)
            self.canvas_GV.leftMouseButtonPressed.connect(self.circ_roi_left_click)
            self.canvas_GV.mouseMoved.connect(self.circ_roi_mouse_move)
        else:
            self.canvas_GV.setMouseTracking(False)
            self.canvas_GV.leftMouseButtonPressed.disconnect()
            self.canvas_GV.mouseMoved.disconnect()
            self.create_img_vis(update=True)

    def define_rect_roi(self):
        if self.rectangle_ROI_BTN.isChecked():
            self.canvas_GV.setMouseTracking(True)
            self.canvas_GV.leftMouseButtonPressed.connect(self.rect_roi_left_click)
            # self.canvas_GV.mouseMoved.connect(self.rect_roi_mouse_move)
        else:
            self.canvas_GV.setMouseTracking(False)
            self.canvas_GV.leftMouseButtonPressed.disconnect()
            self.canvas_GV.mouseMoved.disconnect()
            self.create_img_vis(update=True)

    def disp_membrane_clicked(self):
        self.disp_membrane = self.disp_membrane_BTN.isChecked()
        self.create_img_vis(update=True)

    def disp_seeds_clicked(self):
        self.disp_seeds = self.disp_seeds_BTN.isChecked()
        self.create_img_vis(update=True)

    def segment_img(self, twoD=True):
        self.statusbar.showMessage('Segmenting img')
        self.disp_membrane_BTN.setChecked(True)
        self.disp_membrane = True
        self.segmentator.segment(self.actual_idx, update_fcn=self.update_segmentation, progress_fig=False)
        self.statusbar.showMessage('Segmentation done')

    def center(self):
        screen = QtGui.QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((screen.width() - size.width()) / 2, (screen.height() - size.height()) / 2)

    def wheelEvent(self, event):
        x = int(event.delta() / 120)
        if self.circle_ROI_BTN.isChecked():
            self.circ_roi_radius_SB.setValue(self.circ_roi_radius_SB.value() + x)
            self.circ_roi['radius'] = self.circ_roi_radius_SB.value()
            self.create_img_vis(update=True)

        elif self.eraser_BTN.isChecked():
            self.eraser_roi_radius_SB.setValue(self.eraser_roi_radius_SB.value() + x)
            self.eraser_roi['radius'] = self.eraser_roi_radius_SB.value()
            self.create_img_vis(update=True)

        elif self.threshold_BTN.isChecked():
            self.threshold_SB.setValue(self.threshold_SB.value() + x)

        elif self.marking:
            self.linewidth_SB.setValue(self.linewidth_SB.value() + x)

        else:  # not self.marking:
            if x > 0:
                idx = min(self.actual_idx + 1, self.data.n_slices - 1)
            elif x< 0:
                idx = max(0, self.actual_idx - 1)
            self.slice_SB.setValue(idx)

    def marking_left_click(self, x, y):
        self.last_pt = (int(x), int(y))

    def marking_mouse_move(self, x, y, btn):
        pt = (int(x), int(y))
        linewidth = self.linewidth_SB.value()
        cv2.line(self.data.seeds[self.actual_idx, ...], self.last_pt, pt, self.seed_lbl, linewidth)
        self.last_pt = pt
        self.create_img_vis(update=True)

    def circ_roi_left_click(self, x, y):
        self.circle_ROI_BTN.setChecked(False)

        #update roi
        circ = cv2.circle(np.zeros(self.data.image[self.actual_idx, ...].shape, dtype=np.uint8),
                          self.circ_roi['center'], self.circ_roi['radius'], 1, -1)
        # self.data.roi *= circ
        self.data.circ_roi[self.actual_idx, ...] = circ
        self.data.update_roi()

        self.define_circle_roi()
        self.disp_roi_BTN.setChecked(True)
        if not self.disp_roi_BTN.isChecked():
            self.disp_roi_clicked()
        self.create_img_vis(update=True)

    def circ_roi_mouse_move(self, x, y, btn):
        x = int(round(x))
        y = int(round(y))
        self.circ_roi['center'] = (x, y)
        self.create_img_vis(update=True)

    def rect_roi_left_click(self, x ,y):
        x = int(round(x))
        y = int(round(y))
        self.rect_roi_pts_clicked += 1
        if self.rect_roi_pts_clicked == 1:
            self.rect_roi['pt1'] = (x, y)
            self.canvas_GV.mouseMoved.connect(self.rect_roi_mouse_move)
        else:
            self.rect_roi_pts_clicked += 1
            self.rect_roi['pt2'] = (x, y)
            self.rectangle_ROI_BTN.setChecked(False)

            # update roi
            rect = cv2.rectangle(np.zeros(self.data.image[self.actual_idx, ...].shape, dtype=np.uint8),
                                 self.rect_roi['pt1'], self.rect_roi['pt2'], 1, -1)
            self.data.rect_roi[self.actual_idx, ...] = rect
            self.data.update_roi()

            self.define_rect_roi()
            self.disp_roi_BTN.setChecked(True)
            if not self.disp_roi_BTN.isChecked():
                self.disp_roi_clicked()
            self.create_img_vis(update=True)

    def rect_roi_mouse_move(self, x, y, btn):
        x = int(round(x))
        y = int(round(y))
        self.rect_roi['pt2'] = (x, y)
        self.create_img_vis(update=True)

    def set_marking(self, value):
        self.marking = value
        if self.marking:
            self.canvas_GV.setCursor(QtCore.Qt.CrossCursor)
        else:
            self.canvas_GV.setCursor(QtCore.Qt.ArrowCursor)

    def mark_seeds(self, type):
        self.disp_seeds_BTN.setChecked(True)
        self.disp_seeds = True
        if type == 'o':
            btn = self.mark_obj_BTN
            self.mark_bgd_BTN.setChecked(False)
            self.pen_color = OBJ_COLOR
            self.seed_lbl = OBJ_SEED_LBL
            # self.marking = True
            self.set_marking(True)
            self.canvas_GV.leftMouseButtonPressed.connect(self.marking_left_click)
            self.canvas_GV.mouseMoved.connect(self.marking_mouse_move)
        elif type == 'b':
            btn = self.mark_bgd_BTN
            self.mark_obj_BTN.setChecked(False)
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

        self.segmentator.data = self.data

        # scrolbar update
        self.slice_SB.setMaximum(self.data.n_slices - 1)
        # self.min_slice_idx_LBL.setText(str(1))
        self.max_slice_LBL.setText(str(self.data.n_slices))
        self.slice_SB_changed(0)
        self.set_img_vis(self.data.image[self.actual_idx, ...])

    def create_img_vis(self, img=None, update=True, circ=None):
        if img is None:
            self.image_vis = cv2.cvtColor(self.data.image[self.actual_idx, ...], cv2.COLOR_GRAY2RGB)
        else:
            self.image_vis = img
        # im1 = self.image_vis.copy()
        # overlay = cv2.addWeighted(self.)

        if self.disp_roi_BTN.isChecked():
            roi = self.image_vis.copy()
            roi[np.nonzero(self.data.roi[self.actual_idx, ...])] = ROI_COLOR
            # roi[np.nonzero(self.data.thresh_roi[self.actual_idx, ...])] = ROI_COLOR
            self.image_vis = cv2.addWeighted(roi, ROI_ALPHA, self.image_vis, 1 - ROI_ALPHA, 0)

        if self.disp_seeds:
            self.image_vis[np.nonzero(self.data.seeds[self.actual_idx, ...] == OBJ_SEED_LBL)] = OBJ_COLOR
            self.image_vis[np.nonzero(self.data.seeds[self.actual_idx, ...] == BGD_SEED_LBL)] = BGD_COLOR

        if self.disp_membrane:
            # self.image_vis[np.nonzero(self.data.segmentation[self.actual_idx, ...])] = MEMBRANE_COLOR
            memb = self.image_vis.copy()
            memb[np.nonzero(self.data.segmentation[self.actual_idx, ...])] = MEMBRANE_COLOR
            self.image_vis = cv2.addWeighted(memb, MEMBRANE_ALPHA, self.image_vis, 1 - MEMBRANE_ALPHA, 0)
        # self.image_vis[np.nonzero(self.data.seeds[self.actual_idx, ...])] = [255, 0, 0]

        # if circ is not None:
        #     cv2.circle(self.image_vis, circ[:2], circ[2], CIRC_ROI_COLOR, 1)
        if self.circle_ROI_BTN.isChecked():
            cv2.circle(self.image_vis, self.circ_roi['center'], self.circ_roi['radius'], MARKING_ROI_COLOR, 1)

        if self.rectangle_ROI_BTN.isChecked():
            cv2.rectangle(self.image_vis, self.rect_roi['pt1'], self.rect_roi['pt2'], MARKING_ROI_COLOR, 1)

        # drawing eraser ROI
        if self.eraser_BTN.isChecked():
            cv2.circle(self.image_vis, self.eraser_roi['center'], self.eraser_roi['radius'], ERASER_COLOR, 1)

        # converting image to QImage
        self.qimage = QtGui.QImage(self.image_vis.data, self.image_vis.shape[1], self.image_vis.shape[0], QtGui.QImage.Format_RGB888)

        if update:
            self.canvas_GV.setImage(self.qimage)

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