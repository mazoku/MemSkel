from __future__ import division
__author__ = 'Ryba'

# TODO: pomoci oznaceni BGD "odstrihnout" oblasti, kam se algoritmus nema dostat

# py2exe and libtiff stuff
import sys
sys.path += ['.']
# from ctypes import util
#---------------------------

import Tkinter as tk
# import ttk
import os
import tkFileDialog
import numpy as np

# import matplotlib.pyplot as plt
# import pymorph as pm

# from pylab import *
# TODO: specifikovat potrebne importy z pylabu
import matplotlib.pyplot as plt
from pylab import Circle, Axes, figaspect

from PIL import Image, ImageTk #, PngImagePlugin, TiffImagePlugin
# Image._initialized = 2  # otherwise PIL cannot identify image file after py2exe debuging
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# from scipy.sparse.csgraph import _validation #otherwise could not be compiled by py2exe
from scipy import interpolate

import skimage.exposure as skiexp
import skimage.morphology as skimor

import MySketcher
import memskel_tools as mt
from libtiff import TIFFimage #, TIFF
import ProgressMeter as pmeter

import logging


class MemSkel:

    def __init__(self):
        logging.basicConfig(filename='memSkel_debug.log', level=logging.DEBUG)

        self.root = tk.Tk()
        # self.root.iconbitmap('data/icons/multiIcon.xbm')
        program_dir = sys.path[0]
        icon_dir = 'data/icons/multiIcon.png'
        img = ImageTk.PhotoImage(file=os.path.join(program_dir, icon_dir))
        self.root.tk.call('wm', 'iconphoto', self.root._w, img)
        self.root.protocol('WM_DELETE_WINDOW', self.quit)
        self.root.title('MemSkel')
        self.filename = None

        self.canvas = None
        self.sketch = None
        self.scale = None

        self.seeds = None
        self.newbies = None #nove pridane pixely
        self.numframes = None
        self.segmentingStack = False
        self.repairingStack = False
        self.maskThresh = None  # thresholding mask - segmentation is calculated only inside this mask

        self.dispThresh = tk.BooleanVar()
        self.dispMem = tk.BooleanVar()
        self.dispSkel = tk.BooleanVar()
        self.dispSpline = tk.BooleanVar()

        self.imgStack = None  # images to be analysed

        self.createGUI_Vertical()

        # self.filename = 'z:\Work\Python\MemSkel\data\smallStack.tif'
        # im =  Image.open( self.filename )
        # self.img = np.array( im.convert('L') )
        # self.loadImage()
        self.createWelcomeFig()

    def quit(self):
        plt.close('all') #otherwise would not end if there is an active figure
        self.root.destroy()

    def run(self):

        help_message = '''
        TODO:
            *) setup the img size to match the width of figure (toolbar)
        '''

        self.root.mainloop()
        logging.debug('Application ended.')
#        self.sketch = MySketcher.MySketcher( self.img.copy(), self.seeds, self.on_keypress, self.linewidthSpinbox.get() )
#        self.sketch.run()

    def createGUI_Vertical(self):
        curr_r = 0

        # main toolbar ---------------------------------------
        self.toolbar = tk.Frame(self.root, borderwidth=2, relief='raised')

        # BUTTONS FRAME -------------------------------------------------
        btns_frame = tk.Frame(self.toolbar, borderwidth=2, relief=tk.FLAT)  # .pack(side=tk.TOP)
        btns_frame.grid(row=curr_r, column=0, sticky=tk.E + tk.W)
        curr_r += 1

        img1 = Image.open('data/icons/LoadIcon.png')
        self.useImg1 = ImageTk.PhotoImage(img1)
        loadBtn = tk.Button(btns_frame, image=self.useImg1, command=self.loadBtnCallback)
        loadBtn.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.BOTH, expand=1)

        img2 = Image.open('data/icons/SaveIcon.png')
        self.useImg2 = ImageTk.PhotoImage(img2)
        saveBtn = tk.Button(btns_frame, image=self.useImg2, command=self.saveBtnCallback)
        saveBtn.pack(side=tk.LEFT, padx=5, pady=2, fill=tk.BOTH, expand=1)

        # sep = ttk.Separator( self.toolbar, orient=tk.HORIZONTAL )
        # sep.pack( side=tk.TOP, fill=tk.X, padx=5, pady=8 )

        # THRESHOLDING FRAME -------------------------------------------------
        self.threshFrame = tk.LabelFrame(self.toolbar, text='Thresholding', relief=tk.GROOVE)
        self.threshFrame.grid(row=curr_r, column=0, pady=10, sticky=tk.W + tk.E)
        curr_r += 1

        self.threshScale = tk.Scale(self.threshFrame, orient=tk.HORIZONTAL, from_=0, to=100, command=self.thresholding_CB)
        self.threshScale.pack(side=tk.RIGHT, padx=5, pady=2, fill=tk.BOTH, expand=1)

        # SEGMENTATION FRAME -------------------------------------------------
        segFrame = tk.LabelFrame(self.toolbar, text='Segmentation', relief=tk.GROOVE)
        segBtnsFrame = tk.Frame(segFrame, borderwidth=2, relief=tk.FLAT)
        img3 = Image.open('data/icons/bacteria-icon2.png')
        self.useImg3 = ImageTk.PhotoImage(img3)
        self.segmentFrameBtn = tk.Button(segBtnsFrame, image=self.useImg3, command=self.segmentFrameBtnCallback, state=tk.DISABLED)
        self.segmentFrameBtn.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, padx=5, pady=2)

        img4 = Image.open('data/icons/bacteria-stack.png')
        self.useImg4 = ImageTk.PhotoImage(img4)
        self.segmentStackBtn = tk.Button(segBtnsFrame, image=self.useImg4, command=self.segmentStackBtnCallback, state=tk.DISABLED)
        self.segmentStackBtn.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, padx=5, pady=2)

        segBtnsFrame.pack(side=tk.TOP, fill=tk.X)

        # radiobuttons
        self.markingRB = tk.IntVar()
        self.markingRB.set(1)
        # self.markingRB = 1
        segRBtnsFrame = tk.Frame(segFrame, borderwidth=2, relief=tk.FLAT)
        objRB = tk.Radiobutton(segRBtnsFrame, text='obj', variable=self.markingRB, value=1, indicatoron=0)#, command=lambda: sys.stdout.write(str(self.markingRB.get()) + '\n'))
        bgdRB = tk.Radiobutton(segRBtnsFrame, text='bgd', variable=self.markingRB, value=2, indicatoron=0)#, command=lambda: sys.stdout.write(str(self.markingRB.get()) + '\n'))
        # objRB = tk.Radiobutton(segRBtnsFrame, text='obj', variable=self.markingRB, value=1, indicatoron=0, command=lambda: sys.stdout.write(self.markingRB + '\n'))
        # bgdRB = tk.Radiobutton(segRBtnsFrame, text='bgd', variable=self.markingRB, value=2, indicatoron=0, command=lambda: sys.stdout.write(self.markingRB + '\n'))
        objRB.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, padx=5, pady=2)
        bgdRB.pack(side=tk.LEFT, fill=tk.BOTH, expand=1, padx=5, pady=2)
        segRBtnsFrame.pack(side=tk.TOP, fill=tk.X)

        # linewidth frame ---------------------------------------
        linewidthFrame = tk.Frame(segFrame, relief=tk.GROOVE)
        label = tk.Label(linewidthFrame, text='Linewidth:', relief=tk.FLAT, anchor=tk.W)
        label.pack(side=tk.LEFT, fill=tk.X, expand=1)
        maxLineWidth = 20
        self.linewidthSpinbox = tk.Spinbox(linewidthFrame, from_=1, to=maxLineWidth, width=2, command=self.setLinewidth)
        self.linewidthSpinbox.delete(0, 'end')
        initLinewidth = 5
        self.linewidthSpinbox.insert(0, initLinewidth)
        self.linewidthSpinbox.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=2)

        linewidthFrame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=1)

        segFrame.grid(row=curr_r, column=0, pady=10, sticky=tk.W + tk.E)
        curr_r += 1

        # SKEL FRAME ---------------------------------------
        skelFrame = tk.LabelFrame(self.toolbar, text='Skeleton', relief=tk.GROOVE)
        imgSkel = Image.open('data/icons/skel.png')
        self.useImgSkel = ImageTk.PhotoImage(imgSkel)
        self.skelBtn = tk.Button(skelFrame, image=self.useImgSkel, command=self.skelBtnCallback, state=tk.DISABLED)
        self.skelBtn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)
        skelFrame.grid(row=curr_r, column=0, pady=10, sticky=tk.W + tk.E)
        curr_r += 1

        # APPROXIMATION FRAME ---------------------------------------
        approxFrame = tk.LabelFrame(self.toolbar, text='Approximation', relief=tk.GROOVE)
        img6 = Image.open('data/icons/run.png')
        self.useImg6 = ImageTk.PhotoImage(img6)
        self.splineApproxBtn = tk.Button(approxFrame, image=self.useImg6, command=self.splineApproxBtnCallback, state=tk.DISABLED)
        self.splineApproxBtn.pack(side=tk.TOP, fill=tk.X, padx=5)

        smoothFacFrame = tk.Frame(approxFrame, relief=tk.GROOVE)
        label = tk.Label(approxFrame, text='Smoothing factor:', relief=tk.FLAT, anchor=tk.W)
        label.pack(side=tk.LEFT, fill=tk.X, expand=1)
        maxSmoothFactor = 10
        self.smoothingFactorSpinbox = tk.Spinbox(approxFrame, from_=0, to=maxSmoothFactor, width=2, command=self.splineApproxBtnCallback)#, state=tk.DISABLED )
        self.smoothingFactorSpinbox.delete(0,'end')
        initSmoothFactor = 1
        self.smoothingFactorSpinbox.insert(0, initSmoothFactor)
        self.smoothingFactorSpinbox.pack(side=tk.RIGHT, fill=tk.X, padx=5, pady=5)
        smoothFacFrame.pack(side=tk.TOP, fill=tk.X, padx=5)

        approxFrame.grid(row=curr_r, column=0, pady=10, sticky=tk.W + tk.E)
        curr_r += 1

        #ERASER FRAME --------------------------------------------------------
        eraserFrame = tk.LabelFrame(self.toolbar, text='Eraser', relief=tk.GROOVE)

        img5 = Image.open('data/icons/eraser.png')
        self.useImg5 = ImageTk.PhotoImage(img5)
        self.eraserBtn = tk.Button(eraserFrame, image=self.useImg5, command=self.eraserBtnCallback, state=tk.DISABLED)
        self.eraserBtn.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # radius frame
        erasRadFrame = tk.Frame(eraserFrame, relief=tk.GROOVE)
        label2 = tk.Label(erasRadFrame, text='Radius:', relief=tk.FLAT, anchor=tk.W)
        label2.pack(side=tk.LEFT, fill=tk.X, expand=1)
        minEraserRadius = 2
        maxEraserRadius = 20
        self.eraserRadiusSpinbox = tk.Spinbox(erasRadFrame, from_=minEraserRadius, to=maxEraserRadius, width=2, command=self.setEraserRadius)
        self.eraserRadiusSpinbox.pack(side=tk.LEFT, fill=tk.X, padx=5, pady=2)
        erasRadFrame.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

        eraserFrame.grid(row=curr_r, column=0, pady=10, sticky=tk.W + tk.E)
        curr_r += 1

        # DISPLAY FRAME ---------------------------------------
        dispFrame = tk.LabelFrame(self.toolbar, text='Display', relief=tk.GROOVE)
        threshdispCHB = tk.Checkbutton(dispFrame, text='Thresh', variable=self.dispThresh, command=self.checked, indicatoron=0)
        threshdispCHB.pack(side=tk.LEFT, fill=tk.X, pady=5)
        memdispCHB = tk.Checkbutton(dispFrame, text='Memb', variable=self.dispMem, command=self.checked, indicatoron=0)
        memdispCHB.pack(side=tk.LEFT, fill=tk.X, pady=5)
        skeldispCHB = tk.Checkbutton(dispFrame, text='Skel', variable=self.dispSkel, command=self.checked, indicatoron=0)
        skeldispCHB.pack(side=tk.LEFT, fill=tk.X, pady=5)
        splinedispCHB = tk.Checkbutton(dispFrame, text='Approx', variable=self.dispSpline, command=self.checked, indicatoron=0)
        splinedispCHB.pack(side=tk.LEFT, fill=tk.X, pady=5)

        dispFrame.grid(row=curr_r, column=0, pady=10, sticky=tk.W + tk.E)
        curr_r += 1

        self.dispThresh.set(True)
        self.dispMem.set(True)
        self.dispSkel.set(True)
        self.dispSpline.set(True)

        self.toolbar.pack(side=tk.LEFT, fill=tk.Y)

        # statusbar ---------------------------------------
        self.sbframe = tk.Frame(self.root)
        self.statusbar = tk.Label(self.sbframe, text='This is a status bar', bd=1, anchor=tk.W)
        self.statusbar.pack(side=tk.LEFT, fill=tk.X)
        self.sbframe.pack(side=tk.BOTTOM, fill=tk.X)

    def checked(self):
        if self.sketch:
            self.sketch.redrawFig()

    def skelBtnCallback(self):
        curr = self.sketch.currFrameIdx
        for slc in range( self.numframes ):
            self.sketch.currFrameIdx = slc
            self.skel[slc,:,:] = self.getSkel()
        self.sketch.currFrameIdx = curr
        self.sketch.redrawFig()

    def setLinewidth(self):
        if self.sketch:
            self.sketch.lineWidth = int(self.linewidthSpinbox.get())

    def setEraserRadius(self):
        if self.sketch:
            self.sketch.eraserR = int(self.eraserRadiusSpinbox.get())

    def loadBtnCallback(self):
        self.filename = tkFileDialog.askopenfilename(filetypes = [('image files', '.tiff .tif .png'), ('all files', '.*')])
        if self.filename:
            self.loadImage()
        else:
            self.statusbar.config( text='File not loaded' )
        # print self.filename

    def segmentStackBtnCallback(self):
        if not self.segmentingStack: #1. phase = segmentation of frame
            self.segmentingStack = True

            for i in range(self.sketch.currFrameIdx, self.sketch.numFrames):
                self.sketch.resetFrame(i)

            #self.scale.set( startIdx )
            self.sketch.unregisterCallbacks(mode='view')
            self.sketch.registerCallbacks(mode='mark')
        else: #2. phase = segmentation of the rest of the stack
            self.sketch.unregisterCallbacks(mode='view')
            self.canvas.get_tk_widget().config(cursor='watch')
            self.segmentX2end(self.sketch.currFrameIdx+1)
            self.canvas.get_tk_widget().config(cursor='arrow')
            # self.sketch.registerCallbacks(mode='mark')
            self.sketch.registerCallbacks(mode='view')
            self.segmentingStack = False

#            self.repairStackBtn.config( state=tk.ACTIVE )
            #self.splineApproxBtn.config( state=tk.ACTIVE )
            self.smoothingFactorSpinbox.config(state=tk.NORMAL)

    def segmentFrameBtnCallback(self):
        self.sketch.resetFrame()
        self.sketch.unregisterCallbacks(mode='view')
        self.sketch.registerCallbacks(mode='mark')

    def eraserBtnCallback(self, pos=(-1, -1)):
        self.sketch.unregisterCallbacks(mode='view')
        self.sketch.registerCallbacks(mode='mark')
        # self.keypressCID = self.fig.canvas.mpl_connect( 'key_press_event', self.on_keypress )
        self.sketch.isErasing = True
        if self.sketch.eraser:
            self.sketch.eraser.remove()
        self.sketch.eraser = Circle((pos[0], pos[1]), self.sketch.eraserR, color='b', fill=False, linewidth=5)
        self.sketch.ax.add_artist(self.sketch.eraser)

    def loadImage(self):
        if isinstance(self.filename, str) or isinstance(self.filename, unicode):
            logging.debug('Try to open file: ' + self.filename)
            #self.imgStack = tf.TiffFile(self.filename).asarray().astype( np.uint8)
            self.imgStack = self.pilImg2npArray(Image.open(self.filename))
            self.maskThresh = np.zeros(self.imgStack.shape)
            min_I = self.imgStack.min()
            max_I = self.imgStack.max()
            self.threshScale.configure(from_=min_I)
            self.threshScale.configure(to=max_I)

            # im = MultiImage(self.filename)
            # self.imgStack = im.concatenate()
            self.numframes = self.imgStack.shape[0]

            self.imIdx = 0
            self.img = self.imgStack[self.imIdx]
            self.seeds = np.zeros_like(self.imgStack, np.int32)
            self.skel = np.zeros_like(self.imgStack, np.int32)

            self.approxSkel = list()
            for i in range(self.numframes):
                self.approxSkel.append(np.zeros((1, 2), dtype=np.int)) #np.zeros_like( self.skel, dtype=np.bool)

            # scaler ---------------------------------------
            if self.scale:
                self.scale.destroy()
            self.scale = tk.Scale(self.root, orient=tk.VERTICAL, from_=self.numframes-1, to=0, command=self.scaleCB)
#            self.scale = tk.Scale(self.root, orient=tk.VERTICAL, from_=0, to=self.numframes-1, command=self.scaleCB)
            self.scale.pack(side=tk.RIGHT, fill=tk.Y)

            self.seeds = np.zeros_like(self.imgStack, dtype=np.int32)

            self.createCanvas()
            self.sketch = MySketcher.MySketcher(self.fig, self.ax, self.imgStack, self.seeds, self.on_keypress,
                                                int(self.linewidthSpinbox.get()), int(self.eraserRadiusSpinbox.get()), self)
            self.sketch.run()

            self.segmentFrameBtn.config(state=tk.ACTIVE)
            self.eraserBtn.config(state=tk.ACTIVE)
            self.skelBtn.config(state=tk.ACTIVE)
            self.splineApproxBtn.config(state=tk.ACTIVE)
            if self.numframes > 1:
                self.segmentStackBtn.config(state=tk.ACTIVE)

            self.maskThresh = self.thresh()

            self.statusbar.config(text='File loaded succesfully.')

        else:
            self.statusbar.config(text='Unknown image format')
            return

    def pilImg2npArray(self, im):
        page = 0
        try:
            while 1:
                im.seek(page)
                page = page + 1
        except EOFError:
            pass

        numFrames = page
        im.seek(0)
        imArray = np.zeros((numFrames, np.array(im).shape[0], np.array(im).shape[1]))
        for i in range(numFrames):
            im.seek(i)
            imArray[i, :, :] = np.array(im.convert('L'))
        return imArray

    def npArray2pilImg(self, arr):
        pass

    def scaleCB(self, value):
        self.sketch.setImgFrame(int(value))

    def createWelcomeFig(self):
        filename = 'data/icons/kky.png'
        im =  Image.open(filename)
        img = np.array(im)
        w, h = figaspect(img)
        self.fig = plt.figure(figsize=(w, h))
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        plt.hold(True)
        self.ax = Axes(self.fig, [0, 0, 1, 1], yticks=[], xticks=[], frame_on=False)
        self.fig.delaxes(plt.gca())
        self.fig.add_axes(self.ax)
        plt.imshow(img, aspect='equal')
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def createCanvas(self):
        self.w, self.h = figaspect(self.img)
        self.fig = plt.figure(figsize=(self.w, self.h))
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()

        plt.gray()
        plt.hold(True)
        self.ax = Axes(self.fig, [0, 0, 1, 1], yticks=[], xticks=[], frame_on=False)
        self.fig.delaxes(plt.gca())
        self.fig.add_axes(self.ax)
        plt.imshow(self.img, aspect='equal')
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    def saveBtnCallback(self):
        self.statusbar.config(text='Save button pressed')
        filename = tkFileDialog.asksaveasfilename()
        name, ext = os.path.splitext(filename)

        # writing spline format
        filename_spl = name + '.shapes'
        f_spl = open(filename_spl, 'w')
        f_spl.write('Groups=,\nColors=,\nSupergroups=,\nSupercolors=,\n\n')
        for slc in range(self.numframes):
            tck = self.splines[slc][0]
            knots = tck[0]
            coeffs = tck[1]
            #incrementing coordinates - matlab is indexing from 1 and not from 0
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
        for slc in range(self.numframes):
            # self.sketch.mask[slc, :, :] = pm.close(self.sketch.mask[slc, :, :], np.ones((3,3), dtype=np.bool))
            self.sketch.mask[slc, :, :] = skimor.binary_closing(self.sketch.mask[slc, :, :], selem=np.ones((3,3), dtype=np.bool))
        filename_mem = name + '.tif'
        saveim = np.where(self.sketch.mask, 255, 0)
        tiff = TIFFimage(saveim.astype(np.uint8), description='')
        tiff.write_file(filename_mem, compression='none' )
        del tiff
        # im = Image.fromarray( self.sketch.mask )
        # im.save( filename_mem )
        self.statusbar.config(text=('Results saved to: ' + filename_spl.rsplit('/')[-1] + ', ' + filename_mem.rsplit('/')[-1]))

    def on_keypress(self, event):
        if event.key == 'e':
            #self.eraserBtn.invoke()
            self.eraserBtnCallback((event.xdata, event.ydata))
        if event.key == 'enter':
            if self.sketch.changeMade: #made a change -> recalculate segmentation
                self.segmentFrame()
            elif self.sketch.isErasing:
                self.sketch.isErasing = False
                self.sketch.eraser.remove()
                self.sketch.eraser = None
                if self.skel[self.sketch.currFrameIdx, :, :].any():
                    self.skel[self.sketch.currFrameIdx, :, :] = self.getSkel(show=True)

                self.sketch.unregisterCallbacks(mode='mark')
                self.sketch.registerCallbacks(mode='view')
                self.fig.canvas.draw()

            elif self.segmentingStack:
                self.skel[self.sketch.currFrameIdx, :, :] = self.getSkel(show=True)
                self.segmentStackBtn.invoke()
            else:
                self.sketch.unregisterCallbacks(mode='mark')
                self.sketch.registerCallbacks(mode='view')
                self.fig.canvas.draw()
            # elif not self.sketch.isErasing and self.segmentingStack:
            #     self.eraserBtnCallback((event.xdata, event.ydata))
                # self.sketch.isErasing = True
                # else:
                #     self.sketch.isErasing = False
                #     self.skel[self.sketch.currFrameIdx, :, :] = self.getSkel show=True)
                # if self.segmentingStack:
                #     self.segmentStackBtnCallback()

                    # if not self.skel[self.sketch.currFrameIdx, :, :].any(): #..and skeleton is empty -> get skeleton
                #     self.skel[self.sketch.currFrameIdx, :, :] = self.getSkel(show=True)
                # elif self.segmentingStack: #..and skeleton already computed -> segment stack, if possible
                #     self.segmentStackBtnCallback()
                # else: #segmenting frame and accepting skeleton
                #     self.sketch.unregisterCallbacks(mode='mark')
                #     self.sketch.registerCallbacks(mode='view')
                #
                #     self.splineApproxBtn.config(state=tk.ACTIVE)
                #     self.smoothingFactorSpinbox.config(state=tk.NORMAL)

                # self.sketch.unregisterCallbacks(mode='mark')
                # self.sketch.registerCallbacks(mode='view')
        self.sketch.redrawFig()

    def dataTerm(self, findAll=False):
        seeds = self.sketch.getMarkers() * self.maskThresh[self.sketch.currFrameIdx,:,:]

        if findAll:
            idxs = np.argwhere(self.seeds)
            intensities = np.unique(self.energy[idxs[:, 0],idxs[:, 1]])
            kernel = [0]
            ints = list()
            for i in range(len(intensities)):
                for k in kernel:
                    ints.append(intensities[i] + k)
            ints = np.unique(np.array(ints))

            for i in ints:
                seeds = np.where(self.energy == i, 1, seeds)

        seeds = seeds > 0

        return seeds

    def run_segmentation(self, display=False):
#        ballROut = 1
        ballR = 3
        minDiff = 0.1
        alphaIB = 1

        mask = self.dataTerm(findAll=False)

        strel = np.ones((3, 3), dtype=np.int)

        ballM = np.zeros((2 * ballR + 1, 2 * ballR + 1, 2), dtype=np.int)
        ballM[:, :, 1] = np.tile(np.arange(- ballR, ballR + 1), [2 * ballR + 1, 1])
        ballM[:, :, 0] = ballM[:, :, 1].conj().transpose()

#        ballMOut = np.zeros( (2*ballROut + 1, 2*ballROut + 1, 2), dtype = np.int )
#        ballMOut[:,:,1] = np.tile( np.arange(-ballROut,ballROut+1), [2*ballROut + 1,1] )
#        ballMOut[:,:,0] = ballMOut[:,:,1].conj().transpose()

        self.membraneMask = self.segmentInnerBoundary(self.sketch.img, mask, ballM, strel, minDiff, alphaIB, display)
        # self.membraneMask = pm.close(self.membraneMask)
        self.membraneMask = skimor.binary_closing(self.membraneMask, selem=np.ones((3,3,), dtype=np.bool))
        self.sketch.mask[self.sketch.currFrameIdx, :, :] = self.membraneMask

        self.sketch.redrawFig()

    def segmentInnerBoundary(self, im, mask, ballM, strel, minDiff, alpha, display=False):
        itIB = 0
        drawStep = 5
        maxIterations = 1000
        changeMade = True
        maskThresh = self.maskThresh[self.sketch.currFrameIdx, :, :]
        while changeMade and itIB < maxIterations:
            itIB += 1
            #            mask, accepted, refused = self.iterationIB( im, mask, ballM, strel, minDiff, alpha )
            maskNew, accepted, refused = self.iterationIB2(im, mask.copy(), maskThresh, ballM, strel, minDiff, alpha)
            self.newbies = maskNew - mask
            mask = maskNew
            #            self.newbies = np.zeros( self.newbies.shape, dtype=np.bool )
            #            self.newbies[accepted[0],accepted[1]] = True
            if display and itIB % drawStep == 0:
                self.sketch.mask[self.sketch.currFrameIdx, :, :] = mask
                self.sketch.redrawFig()

            if not accepted.any():
                changeMade = False
        # mask = pm.close(mask, np.ones((5, 5), dtype=np.int))
        mask = skimor.binary_closing(mask, selem=np.ones((5, 5), dtype=np.int))
        return mask

    def iterationIB2(self, im, mask, maskThresh, ballM, strel, minDiff, alpha):
        # dilm = pm.dilate(self.newbies, strel) * maskThresh
        dilm = skimor.binary_dilation(self.newbies, strel) * maskThresh
        newbies2 = dilm - (self.newbies * maskThresh)
        newbiesL = np.argwhere(newbies2)#.tolist()

        accepted = list()
        refused = list()

        #        labels = mt.fuseImages( (mask, newbies2), 'wg', type = 'imgs')
        #        plt.figure()
        #        plt.imshow( im ), plt.hold( True )
        #        plt.imshow( labels )
        #        plt.show()

        maskV = np.argwhere(mask)
        maskV = im[maskV[:, 0], maskV[:, 1]]
        meanMask = maskV.mean()

        for i in range(newbiesL.shape[0]):
            newbie = newbiesL[i]
            inners, outers = self.maskNewbie(newbie, mask, ballM.copy())

            distIn, distOut = self.getDistances(im, newbie, inners, outers)
            distMask = np.absolute(meanMask - im[newbie[0], newbie[1]])

            weightDistIn = alpha * distIn + (1 - alpha) * distMask

            if weightDistIn < distOut or np.absolute(distIn - distOut) < minDiff:
                mask[newbie[0], newbie[1]] = True
                accepted.append(newbie)
            else:
                refused.append(newbie)

        accepted = np.array(accepted)
        refused = np.array(refused)

        return mask, accepted, refused

    def maskNewbie(self, newbie, mask, ballM):
        ballR = int(ballM.shape[0] / 2)
        ballM[:, :, 0] += newbie[0]
        ballM[:, :, 1] += newbie[1]

        #    warnings.warn( 'Place for possible speeding up the process.' )
        #    if np.amin( ballM[:,:,0] ) < 0:
        try:
            while np.amin(ballM[:, :, 0]) < 0:
                ballM = ballM[1:, :, :]
            while np.amin(ballM[:, :, 1]) < 0:
                ballM = ballM[:, 1:, :]
            while np.amax(ballM[:, :, 0]) >= mask.shape[0]:
                ballM = ballM[:-1, :, :]
            while np.amax(ballM[:, :, 1]) >= mask.shape[1]:
                ballM = ballM[:, :-1, :]
        except ValueError:
            self.statusbar.config(text='Error while controling ball boundaries.')

        masked = mask[ballM[:, :, 0], ballM[:, :, 1]]
        innersI = np.nonzero(masked)
        masked[ballR, ballR] = True #exclude center from computation
        outersI = np.nonzero(np.invert(masked))

        inners = ballM[innersI[0], innersI[1], :]
        outers = ballM[outersI[0], outersI[1], :]

        #    return masked, ballM
        return inners, outers

    def getDistances(self, im, newbie, inners, outers):
        intIn = im[inners[:, 0], inners[:, 1]]
        intOut = im[outers[:,0], outers[:, 1]]
        intNew = im[newbie[0], newbie[1]]

        meanIn = intIn.mean()
        meanOut = intOut.mean()

        distIn = np.absolute(meanIn - intNew)
        distOut = np.absolute(meanOut - intNew)

        return distIn, distOut

    def getSkel(self, show=False, data='mask'):
        # extracting skelet of membrane
        if data == 'mask':
            data = self.sketch.mask[self.sketch.currFrameIdx, :, :]
        elif data == 'spline':
            data = self.approxSkel[self.sketch.currFrameIdx, :, :]
        # memskel = pm.thin(data)
        memskel = skimor.medial_axis(data)
        # processing skelet - removing not-closed parts
        memskel = mt.sequentialThinning(memskel, type='golayE')

        if show:
            # display skelet
            #labels = mt.fuseImages((self.membraneMask, pm.thin(self.membraneMask) - memskel, memskel), 'wrb')
            # labels = mt.fuseImages((pm.thin(data) - memskel, memskel), 'rb')
            labels = mt.fuseImages((skimor.medial_axis(data) - memskel, memskel), 'rb')

#            f = plt.figure(figsize=(self.w,self.h))
#            f.ax = Axes(f, [0,0,1,1], yticks=[], xticks=[], frame_on=False)
#            f.delaxes(plt.gca())
#            f.add_axes(f.ax)
#            plt.imshow(self.img, aspect='equal'), plt.hold(True)
            plt.imshow(labels)
            self.canvas.draw()
#            plt.show()

        #creating image to save
#        im = self.img.copy()
#        if len(im.shape) == 2:
#            im = cv2.cvtColor( im, cv2.COLOR_GRAY2RGB )
#        imP = np.argwhere(memskel)
#        im[ imP[:,0], imP[:,1], : ] = [255,0,0]
#        im = cv2.cvtColor( im, cv2.COLOR_RGB2BGR )
#
#        fn = 'z:/Work/Bunky/images/skel/skelOverlay_' + string.zfill(idx,3)  + '.png'
#        cv2.imwrite( fn, im )

        return memskel

    def thresholding_CB(self, value):
        self.statusbar.config(text='New threshold = %i' % int(value))
        self.thresh(t=int(value))
        if self.sketch is not None:
            self.sketch.redrawFig()

    def thresh(self, pt=70, t=None):
        '''
        Thresholding - default method is percentile.
        @param pt: percentile threshold, default is 70%
        @type pt: int
        @param t: threshold value, default is None. If it is not None, then this value is used prior to the
        percentile thresholding.
        @type t: float
        @return: thresholded image
        @rtype: ndarray, dtype=bool
        '''

        if self.imgStack is not None:
            if t is None:
                # percent thresh
                hist, bins = skiexp.histogram(self.imgStack, nbins=100)
                hist = hist / hist.sum()
                # hist = hist.astype(np.float) / hist.sum()
                cum_hist = hist.copy()
                for i in range(1, len(cum_hist)):
                    cum_hist[i] += cum_hist[i - 1]

                diff = cum_hist - (pt / 100)
                diff *= diff > 0

                t_ind = np.nonzero(diff)[0][0]
                t = bins[t_ind]
                self.threshScale.set(t)

            self.maskThresh = self.imgStack > t
            return self.maskThresh

    def segmentFrame(self):
        self.newbies = self.sketch.seeds[self.sketch.currFrameIdx, :, :] > 0
        self.sketch.unregisterCallbacks(mode='mark')

        # self.canvas.get_tk_widget().config(cursor='wait')
        self.canvas.get_tk_widget().config(cursor='watch')
        self.run_segmentation(display=True)

        # self.sketch.setInitMask(pm.erode(self.sketch.mask[self.sketch.currFrameIdx, :, :],
        #                                  np.ones((3, 3), dtype=np.bool)), setMask=False)
        self.sketch.setInitMask(skimor.binary_erosion(self.sketch.mask[self.sketch.currFrameIdx, :, :],
                                                      np.ones((3, 3), dtype=np.bool)), setMask=False)
        self.sketch.registerCallbacks(mode='mark')
        self.sketch.changeMade = False

#        self.repairFrameBtn.config( state=tk.ACTIVE )

        self.statusbar.config(text='Frame segmetation finished.')

    def segmentX2end( self, startIdx ):
        self.segmentX2Y( startIdx, self.numframes-1 )

    def segmentX2Y( self, startIdx, endIdx ):
        t = 5 #maximal difference for accepting point as a seed point
        nFrames = endIdx - startIdx + 1 #from 2nd to the 4th makes 3 frames => 3 = 4 - 2 + 1
        # self.progressbar = ttk.Progressbar( self.sbframe, orient=tk.HORIZONTAL, length=200, mode='determinate', maximum=nFrames-1 )
        self.progressbar = pmeter.Meter( self.sbframe, relief=tk.RIDGE, bd=3 )
        self.progressbar.set( 0, '')
        self.progressbar.pack( side=tk.RIGHT, fill=tk.X, padx=10 )

        for i in range( startIdx, endIdx + 1 ):
            self.statusbar.config( text=str('Processing frame #' + str(i+1) + ' / ' + str(endIdx+1) + '...') )
            self.sketch.currFrameIdx = i
            self.scale.set( i )
            #oldskel = self.skel
            self.imIdx = i
            #self.img = self.imMI[self.imIdx]
            self.img = self.imgStack[i,:,:]
            self.sketch.img = self.img
            self.energy = self.img
#            diffim = cv2.absdiff( self.img, self.imgStack[i-1,:,:] )
            diffim = mt.absdiffNP( self.img, self.imgStack[i-1,:,:] )
            mask = ( diffim < t ) * self.skel[i-1,:,:] > 0
            self.sketch.setInitMask( mask )
            self.newbies = mask
            #self.run_segmentation(display=False)
            self.run_segmentation( display=True )
            self.skel[i,:,:] = self.getSkel( show=False )

            if not self.skel[i,:,:].any() or self.skel[i,:,:].sum() < 10:
#                print 'starting over...'
                self.sketch.setInitMask( self.skel[i-1,:,:] )
                self.run_segmentation( display=True )
                self.skel[i,:,:] = self.getSkel( show=False )

            #            fn = 'z:/Work/Bunky/images/skel/skel_' + string.zfill(i,3)  + '.png'
            #            cv2.imwrite( fn, 255*self.skel )
            #            print 'done\n'
            #self.progressbar.step()
            self.progressbar.step( float(i)/endIdx )
            self.canvas.draw()

        self.progressbar.destroy()

        self.sketch.redrawFig()
        self.statusbar.config( text='Process completed.' )

    def splineApproxBtnCallback(self):
        #splinesL = list()
        pathsL = list()

        self.splines = list()
        self.approxSkel = list()
        for i in range(self.numframes):
            self.approxSkel.append(np.zeros((1, 2),dtype=np.int))
            self.splines.append(np.zeros((1, 2),dtype=np.int))

        nProcessedFrames = np.amax(np.amax(self.skel, axis=1), axis=1).sum()
        m = self.skel.sum() / nProcessedFrames #average number of points in skelet
        maxsf = m + np.sqrt(2 * m) #maximal recommended smoothing factor according to scipy documentation
        sfLevel = int(self.smoothingFactorSpinbox.get())
        nLevels = 10 #number of smoothing levels
        sf = int(sfLevel * ((2 * maxsf) / nLevels))

        for i in range(self.numframes):
            if not self.skel[i, :, :].any():
                continue
            path = mt.skel2path(self.skel[i, :, :].copy())
            pathsL.append(path)
            x = []
            y = []
            for point in path:
                x.append(point[0])
                y.append(point[1])
            # x.append( x[0] )
            # y.append( y[0] )

            #number of knots
            # nSkelPts = len(x)
            # N = nSkelPts / 2.
            # u = np.arange(0, N)
            # u = np.linspace(0, 1, N)

            tcku, fp, ier, msg = interpolate.splprep((x,y), s=sf, full_output=1, per=1)
            # tcku, fp, ier, msg = interpolate.splprep((x,y), s=sf, t=N, task=-1, full_output=1, per=0)
            #splinesL.append(tcku)
            # u = np.arange(0,1.01,0.01)
            u = tcku[1]

            #convert coefficients to integers
            # tcku[0][1][0] = (tcku[0][1][0]).astype(np.int)
            # tcku[0][1][1] = (tcku[0][1][1]).astype(np.int)

            self.splines[i] = tcku
            self.approxSkel[i] = np.array(interpolate.splev(x=u, tck=tcku[0]))
#            ptsAS = np.array(interpolate.splev(x=u, tck=tcku[0]))
            #ptsAS = np.round(ptsAS).astype('uint8')

#            im = np.zeros_like(self.approxSkel[i, :, :], dtype=np.bool)
#            im[ptsAS[0, :], ptsAS[1, :]] = True
#            self.approxSkel[i, :, :] = im
            #self.approxSkel[i, :, :] = self.getSkel(data='spline')

        self.sketch.redrawFig()

#            print msg

#            tck = tcku[0]
#            knots = tck[0]
#            coeffs = tck[1]
#            deg = tck[2]
#            u = tcku[1]
#
#            print('number of points: ' + str(len(x)))
#            print('number of knots: ' + str(len(knots)))
#            print('number of coefficients: ' + str(u.shape[0]))
#            print('degree of spline: ' + str(deg))

if __name__ == '__main__':
    MemSkel().run()