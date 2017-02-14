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
from PyQt4.QtGui import QMainWindow, QApplication, QPixmap
from memskel_GUI_QT import Ui_MainWindow

import numpy as np
import skimage.io as skiio
import cv2

class MainWindow(QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)

        logo_fname = 'data/icons/kky.png'
        self.img_vis = QPixmap(logo_fname)
        self.canvas_L.setPixmap(self.img_vis)
        

if __name__ == '__main__':
    app = QApplication(sys.argv)
    frame = MainWindow()
    frame.show()
    app.exec_()