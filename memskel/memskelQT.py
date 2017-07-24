from __future__ import division

import os
from libtiff import TIFFimage
# import ProgressMeter as pmeter
# import logging

import sys
from PyQt4 import QtCore, QtGui
from memskel_GUI_QT import Ui_MainWindow

import numpy as np
import cv2
from scipy import interpolate

from imagedata import ImageData
from MyImageViewer import ImageViewerQt
from segmentator import Segmentator
import skimage.morphology as skimor
import pymorph as pm
import cPickle as pickle
# import shelve
import gzip

from constants import *


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
        self.disp_skelet = self.disp_skelet_BTN.isChecked()
        self.disp_approx = self.disp_approximation_BTN.isChecked()
        self.state_fname = 'data/program_state.pklz'
        # self.x

        # OVERRIDING ----
        # self.canvas_L.resized.connect(self.resize_canvas_event)
        # self.canvas_L.resizeEvent = self.resize_canvas_event

        # SIGNALS ----
        self.open_BTN.clicked.connect(self.load_data)
        self.save_BTN.clicked.connect(self.save_data)
        self.slice_SB.valueChanged.connect(self.slice_SB_changed)
        self.mark_obj_BTN.clicked.connect(lambda: self.mark_seeds('o'))
        self.mark_bgd_BTN.clicked.connect(lambda: self.mark_seeds('b'))
        self.segment_img_BTN.clicked.connect(self.segment_img)
        self.segment_stack_BTN.clicked.connect(self.segment_stack)

        self.disp_membrane_BTN.clicked.connect(self.disp_membrane_clicked)
        self.disp_seeds_BTN.clicked.connect(self.disp_seeds_clicked)
        self.disp_roi_BTN.clicked.connect(self.disp_roi_clicked)
        self.disp_skelet_BTN.clicked.connect(self.disp_skelet_clicked)
        self.disp_approximation_BTN.clicked.connect(self.disp_approximation_clicked)

        self.circle_ROI_BTN.clicked.connect(self.define_circle_roi)
        self.circ_roi_radius_SB.valueChanged.connect(self.circ_radius_SB_changed)
        self.rectangle_ROI_BTN.clicked.connect(self.define_rect_roi)

        self.threshold_BTN.clicked.connect(self.threshold_clicked)
        self.threshold_SB.valueChanged.connect(self.threshold_changed)

        self.eraser_BTN.clicked.connect(self.eraser_clicked)
        self.eraser_roi_radius_SB.valueChanged.connect(self.eraser_radius_SB_changed)

        self.skeleton_BTN.clicked.connect(self.skeletonize_clicked)
        self.approximation_BTN.clicked.connect(self.approximation_clicked)

        self.smoothing_fac_SB.valueChanged.connect(self.smoothing_fac_changed)

        self.save_state_BTN.clicked.connect(self.save_state)
        self.load_state_BTN.clicked.connect(self.load_state)

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

    def smoothing_fac_changed(self, value):
        self.approximation_clicked()

    def sequential_thinning(self, skel, type='both'):
        # TODO: is it possible to use skimage.thin to do that? This way we could remove one dependency (pymorph)
        golayL1Hit = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]])
        golayL1Miss = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        golayL = pm.se2hmt(golayL1Hit, golayL1Miss)

        golayE1Hit = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
        golayE1Miss = np.array([[0, 0, 0], [1, 0, 1], [1, 1, 1]])
        golayE = pm.se2hmt(golayE1Hit, golayE1Miss)

        if type in ['golayL', 'both']:
            skel = pm.thin(skel, Iab=golayL, n=-1, theta=45, direction='clockwise')
        if type in ['golayE', 'both']:
            skel = pm.thin(skel, Iab=golayE, n=-1, theta=45, direction='clockwise')
        else:
            raise AttributeError('Wrong type specified, valid types are: golayL, golayE, both')

        # changed = True
        # while changed:
        #    skelT = pm.thin(skel, Iab = golayE, n = 1, theta = 45, direction = "clockwise")
        #    plt.figure()
        #    plt.subplot(121), plt.imshow( skel ), plt.gray()
        #    plt.subplot(122), plt.imshow( skelT ), plt.gray()
        #    plt.show()
        #
        #    if (skel == skelT).all() :
        #        changed = False
        #    skel = skelT

        return skel

    def skeletonize_clicked(self):
        data = self.data.segmentation[self.actual_idx, ...]
        skel = skimor.medial_axis(data)
        # processing skelet - removing not-closed parts
        self.data.skelet[self.actual_idx, :, :] = self.sequential_thinning(skel, type='golayE')
        self.disp_skelet_BTN.setChecked(True)
        self.disp_skelet_clicked()
        self.create_img_vis(update=True)

    def spline_approximation(self):
        pass

    def skel2path(self, skel, startPt=[]):
        nPoints = len(np.nonzero(skel)[0])
        path = np.zeros((nPoints, 2))

        if not startPt:
            startPt = np.argwhere(skel)[0, :]
        skel[startPt[0], startPt[1]] = 0
        path[0, :] = startPt

        nghbsI = np.array(([-1, 0], [0, -1], [0, 1], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]))

        currPt = startPt
        for i in range(1, nPoints - 1):
            mask = np.array([nghbsI[:, 0] + currPt[0], nghbsI[:, 1] + currPt[1]]).conj().transpose()
            nghbs = skel[mask[:, 0], mask[:, 1]]
            firstI = np.argwhere(nghbs)[0]  # get index of first founded neighbor
            currPt = np.squeeze(mask[firstI, :])  # get that neighbor
            path[i, :] = currPt
            skel[currPt[0], currPt[1]] = 0

        path[-1, :] = np.argwhere(skel)  # should be the last remaining pixel

        return path

    def approximation_clicked(self):
        pathsL = []
        # spline = []
        # approx = []
        # skel = self.data.skelet[self.actual_idx, ...]
        # for i in range(self.numframes):
        #     approx.append(np.zeros((1, 2), dtype=np.int))
        #     spline.append(np.zeros((1, 2), dtype=np.int))

        nProcessedFrames = np.amax(np.amax(self.data.skelet, axis=1), axis=1).sum()
        m = self.data.skelet.sum() / nProcessedFrames  # average number of points in skelet
        maxsf = m + np.sqrt(2 * m)  # maximal recommended smoothing factor according to scipy documentation
        sfLevel = int(self.smoothing_fac_SB.value())
        nLevels = 10  # number of smoothing levels
        sf = int(sfLevel * ((2 * maxsf) / nLevels))

        for i in range(self.data.n_slices):
            if not self.data.skelet[i, :, :].any():
                self.data.spline[i] = None
                self.data.approx_skel[i] = None
                continue
            path = self.skel2path(self.data.skelet[i, :, :].copy())
            pathsL.append(path)
            x = []
            y = []
            for point in path:
                x.append(point[1])
                y.append(point[0])

            # tck, u, fp, ier, msg = interpolate.splprep((x, y), s=sf, full_output=1, per=1)
            (tck, u), fp, ier, msg = interpolate.splprep((x, y), s=sf, full_output=1, per=1)
            # x = interpolate.splprep((x, y), s=sf, full_output=1, per=1)
            # u = tcku[1]

            self.data.spline[i] = (tck, u)
            self.data.approx_skel[i] = np.array(interpolate.splev(x=u, tck=tck))

            self.disp_approximation_BTN.setChecked(True)
            self.disp_approximation_clicked()
            self.create_img_vis(update=True)

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

    def disp_skelet_clicked(self):
        self.disp_skelet = self.disp_skelet_BTN.isChecked()
        self.create_img_vis(update=True)

    def disp_approximation_clicked(self):
        # skel = np.load('data/skelet.npy')
        # tck = np.load('data/tck.npy')
        # u = np.load('data/u.npy')
        # self.data.skelet[self.actual_idx, ...] = skel
        # self.data.spline[self.actual_idx] = (tck, u)
        # self.data.approx_skel[self.actual_idx] = np.array(interpolate.splev(x=u, tck=tck))

        self.disp_approx = self.disp_approximation_BTN.isChecked()
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

    def segment_img(self):
        self.statusbar.showMessage('Segmenting img ...')
        self.disp_membrane_BTN.setChecked(True)
        self.disp_membrane = True
        self.segmentator.segment(self.actual_idx, update_fcn=self.update_segmentation, progress_fig=False)
        self.statusbar.showMessage('Segmentation done')
        self.segment_stack_BTN.setEnabled(True)
        self.data.processed[self.actual_idx] = True
        self.disp_membrane_BTN.setChecked(True)

    def segment_stack(self):
        # segmentation
        self.statusbar.showMessage('Segmenting stack ...')

        self.statusbar.showMessage('Segmentation done')

        # skeletonization

        # approximation

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
            self.canvas_GV.leftMouseButtonPressed.connect(self.marking_left_click)
            self.canvas_GV.mouseMoved.connect(self.marking_mouse_move)

        if not btn.isChecked():
            # self.marking = False
            self.set_marking(False)
            self.canvas_GV.leftMouseButtonPressed.disconnect()
            self.canvas_GV.mouseMoved.disconnect()
            return

    def slice_SB_changed(self, value):
        self.actual_slice_LBL.setText(str(value + 1))
        self.actual_idx = value

        self.segment_stack_BTN.setEnabled(self.data.processed[self.actual_idx])

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

    def save_data(self):
        # self.statusbar.config(text='Save button pressed')

        filename = str(QtGui.QFileDialog.getSaveFileName(self, 'Save membrane', 'data/',
                                                         "Image files (*.jpg *.gif *.tiff *.tif)"))
        name, ext = os.path.splitext(filename)

        # writing spline format
        filename_spl = name + '.shapes'
        f_spl = open(filename_spl, 'w')
        f_spl.write('Groups=,\nColors=,\nSupergroups=,\nSupercolors=,\n\n')
        for slc in range(self.data.n_slices):
            if self.data.spline[slc] is not None:
                tck = self.data.spline[slc][0]
                knots = tck[0]
                coeffs = tck[1]
                # incrementing coordinates - matlab is indexing from 1 and not from 0
                for coords in coeffs:
                    coords += 1
                k = tck[2]
                strK = str(k)
                strKnots = ','.join(map(str, knots.tolist()))
                strX = ','.join(map(str, coeffs[0].tolist()))
                strY = ','.join(map(str, coeffs[1].tolist()))

                f_spl.write('type=spline\n')
                f_spl.write('name=\n')
                f_spl.write('group=\n')
                f_spl.write('color=255\n')
                f_spl.write('supergroup=null\n')
                f_spl.write('supercolor=null\n')
                f_spl.write('in slice=' + str(slc) + '\n')
                f_spl.write('k=' + strK + '\n')
                f_spl.write('knots:\n')
                f_spl.write(strKnots + '\n')
                f_spl.write('control points:\n')
                f_spl.write(strX + '\n')
                f_spl.write(strY + '\n\n')

        f_spl.close()

        # saving membrane image
        # for slc in range(self.data.n_slices):
            # self.sketch.mask[slc, :, :] = pm.close(self.sketch.mask[slc, :, :], np.ones((3,3), dtype=np.bool))
            # self.data.segmentation[slc, :, :] = skimor.binary_closing(self.sketch.mask[slc, :, :],
            #                                                     selem=np.ones((3, 3), dtype=np.bool))
        filename_mem = name + '.tif'
        saveim = np.where(self.data.segmentation, 255, 0)
        tiff = TIFFimage(saveim.astype(np.uint8), description='')
        tiff.write_file(filename_mem, compression='none')
        del tiff

    def save_state(self):
        print "Saving program sate into the file '{}'...".format(self.state_fname),
        # out = {'im': self.data.image,
        #        'seeds': self.data.seeds,
        #        'segmentation': self.data.segmentation,
        #        'roi': self.data.roi,
        #        'idx': self.actual_idx}
        state = self.create_program_state()
        with gzip.open(self.state_fname, 'wb') as f:
            pickle.dump(state, f)
        print 'done'

    def load_state(self):
        with gzip.open(self.state_fname, 'rb') as f:
            state = pickle.load(f)
        pass
        self.set_state(state)

    def set_state(self, state):
        self.threshold_SB.setValue(state['threshold'])
        pass

    def create_program_state(self):
        state = {'data': self.data,
                 'idx': self.actual_idx,
                 'sidp_'
                 'disp_seeds': self.disp_seeds_BTN.isChecked(),
                 'disp_membrane': self.disp_membrane_BTN.isChecked(),
                 'disp_skelet': self.disp_skelet_BTN.isChecked(),
                 'disp_approx': self.disp_approximation_BTN.isChecked(),
                 'threshold': self.threshold_SB.value(),
                 'circle_radius': self.circ_roi_radius_SB.value(),
                 'smoothing_factor': self.smoothing_fac_SB.value(),
                 'eraser_radius': self.eraser_roi_radius_SB.value()
                 }
        return state

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

        if self.disp_skelet:
            self.image_vis[np.nonzero(self.data.skelet[self.actual_idx, ...])] = SKEL_COLOR

        if self.disp_approx:
            # self.image_vis[np.nonzero(self.data.skelet[self.actual_idx, ...])] = APPROX_COLOR
            # cv2.polylines(self.image_vis, self.data.approx_skel[self.actual_idx], True, APPROX_COLOR,
            #               thickness=1, lineType=cv2.LINE_AA)
            if self.data.approx_skel[self.actual_idx] is not None:
                pts = self.data.approx_skel[self.actual_idx].astype(np.int32)
                for i in range(pts.shape[1] - 1):
                    cv2.line(self.image_vis, tuple(pts[:, i]), tuple(pts[:, i + 1]), APPROX_COLOR, thickness=1, lineType=cv2.LINE_AA)

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
    fname = 'data/smallStack.tif'
    memskel.load_data(fname=fname)
    memskel.show()
    app.exec_()