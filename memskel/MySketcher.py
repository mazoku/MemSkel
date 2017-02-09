#import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import matplotlib.colors as matcol
#import matplotlib.text as text
import pymorph as pm
from pylab import *
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt
#import cv2
#import cv2.cv as cv
#import myTools as mt
import memskel_tools as mt
import warnings


class MySketcher:
    def __init__( self, fig, ax, imgStack, seeds, on_keypress, linewidth, eraserR, memskel):
        self.imgStack = imgStack
        self.numFrames = self.imgStack.shape[0]
        self.memskel = memskel

        self.currFrameIdx = 0
        self.img = self.imgStack[self.currFrameIdx,:,:]
        self.seeds = seeds

        x = np.arange( 0, self.seeds[self.currFrameIdx,:,:].shape[1] )
        y = np.arange( 0, self.seeds[self.currFrameIdx,:,:].shape[0] )
        self.xgrid, self.ygrid = np.meshgrid( x, y )

        self.fig = fig
        self.ax = ax
#        self.w, self.h = self.img.shape
#        self.fig = plt.figure( figsize=(self.w,self.h) )
#        plt.gray()
#        plt.hold( True )
#        self.ax = Axes( self.fig, [0,0,1,1], yticks=[], xticks=[], frame_on=False )
#        self.fig.delaxes( plt.gca() )
#        self.fig.add_axes( self.ax )
#        plt.imshow( self.img, aspect='equal' )

        self.lineWidth = linewidth
        #self.text = Text( x = 10, y = 10, text = 'linewidth = ' + str(self.lineWidth), color = 'green' )
        #self.ax.add_artist( self.text )
        self.linesL = list()
        self.line = None
        self.lineMarkers = None
        #        self.contours = None
        self.changeMade = False
        self.eraser = None
        self.eraserR = eraserR
        self.maxR = 20
        self.minR = 2
        self.isErasing = False
        self.eraserMask = self.calcEraserMask()

        self.prev_pt = None
        self.mousepressCID = None
        self.mousereleaseCID = None
        self.isMarking = False
        self.mousemoveCID = None
        self.mousescrollCID = None
        self.keypressCID = None
        self.on_keypressM = on_keypress

        self.mask = np.zeros_like( self.seeds, dtype=np.bool )

    def run(self):
        self.registerCallbacks( 'view' )
        #plt.show()

    def on_mousePress( self, event ):
        if not self.isErasing and event.button == 1 and event.inaxes:
            self.isMarking = True
            self.lineMarkers = np.zeros( self.seeds[self.currFrameIdx,:,:].shape, dtype=np.int32 )#np.zeros_like( self.seeds, dtype=np.int32 )
            self.line = Line2D( [round(event.xdata)], [round(event.ydata)], linewidth=self.lineWidth, color='red' )
            self.linesL.append( self.line )
            self.ax.add_artist( self.line )
            self.fig.canvas.draw()
        elif self.isErasing and event.button == 1 and event.inaxes:
            self.eraser = Circle( (event.xdata, event.ydata), self.eraserR, color = 'b', fill = False, linewidth = 5 )
            self.ax.add_artist( self.eraser )
            self.isErasing = True
#            self.text.set_text( 'eraser radius = ' + str(self.eraserR) )
            self.erase( event.xdata, event.ydata )
            self.fig.canvas.draw()

    def on_mouseRelease( self, event ):
        if not self.isErasing and event.button == 1:
            self.isMarking = False
            self.lineMarkers = pm.dilate( self.lineMarkers.astype(np.bool) , np.ones( (self.lineWidth,self.lineWidth), dtype=np.bool ) )
            self.seeds[self.currFrameIdx,:,:] |= np.copy( self.lineMarkers )
        elif self.isErasing and event.button == 1:
            if self.eraser:
                self.eraser.remove()
            self.eraser = Circle( (event.xdata, event.ydata), self.eraserR, color = 'b', fill = False, linewidth = 5 )
            self.ax.add_artist( self.eraser )
            self.fig.canvas.draw()

    def on_mouseMove( self, event):
        if event.inaxes and self.isMarking:
        #            pt = ( round(event.ydata), round(event.xdata) )
            if event.button == 1:
                xd = np.hstack( (self.line.get_xdata(), event.xdata) )
                yd = np.hstack( (self.line.get_ydata(), event.ydata) )
                self.line.set_xdata( xd )
                self.line.set_ydata( yd )
                self.fig.canvas.draw()
                self.lineMarkers[int(event.ydata), int(event.xdata)] = 1
                #                self.dests[1][event.ydata, event.xdata] = 1
                self.changeMade = True
            else:
                self.prev_pt = None
        elif self.isMarking and not event.inaxes:
            self.isMarking = False
            self.lineMarkers = pm.dilate( self.lineMarkers, np.ones( (self.lineWidth,self.lineWidth), dtype=np.int ) )
            self.seeds[self.currFrameIdx,:,:] |= self.lineMarkers
        elif self.isErasing and event.button == 1: #left button is pressed -> erasing
            self.eraser = Circle( (event.xdata, event.ydata), self.eraserR, color = 'r', fill = False, linewidth = 5 )
            self.ax.add_artist( self.eraser )
            self.erase( event.xdata, event.ydata )
        elif self.isErasing and event.button != 1: #left button not pressed -> only positioning
            if self.eraser:
                self.eraser.remove()
            self.eraser = Circle( (event.xdata, event.ydata), self.eraserR, color = 'b', fill = False, linewidth = 5 )
            self.ax.add_artist( self.eraser )

            self.fig.canvas.draw()
        else:
            self.isMarking = False

    def calcEraserMask( self ):
    #        circsL = list()
    #        for r in range( self.maxR + 1 ):
    #            size = 2 * r + 1
    #            circ = np.zeros( (size, size), dtype=np.int )
    #            for i in range(size):
    #                for j in range(size):
    #                    if math.pow(i,2) + math.pow(j,2) <= math.pow(r,2):
    #                        circ[i,j] = 1
    #            circsL.append( np.copy(circ) )
    #        return circsL
        circ = np.ones( (2*self.maxR+1,2*self.maxR+1), dtype=np.uint8  )
        circ[self.maxR,self.maxR] = 0
        circ = distance_transform_edt( circ )
        #circ = cv2.distanceTransform( circ, cv.CV_DIST_L2, 3)

        return circ

    def on_mouseScrollViewMode( self, event):
        #self.currFrameIdx += int( event.step )
        if event.step > 0:
            self.currFrameIdx = min( self.numFrames-1, self.currFrameIdx + event.step )
        else:
            self.currFrameIdx = max( 0, self.currFrameIdx + event.step )
        self.memskel.scale.set( self.currFrameIdx )

        #self.img = self.imgStack[ self.currFrameIdx,:,: ]
        #self.redrawFig()

    def on_mouseScrollMarkMode( self, event):
        if event.step > 0:
            if self.isErasing:
                self.memskel.eraserRadiusSpinbox.invoke( 'buttonup' )
                self.on_mouseMove( event )
                #self.redrawFig()
            else:
                self.memskel.linewidthSpinbox.invoke( 'buttonup' )
        else:
            if self.isErasing:
                self.memskel.eraserRadiusSpinbox.invoke( 'buttondown' )
                self.on_mouseMove( event )
                # self.redrawFig()
            else:
                self.memskel.linewidthSpinbox.invoke( 'buttondown' )

    def setImgFrame( self, idx):
        self.currFrameIdx = idx
        self.img = self.imgStack[ self.currFrameIdx,:,: ]
        self.redrawFig()

    def erase(self, x, y):
        circI = np.argwhere(self.eraserMask <= self.eraserR)
        circI[:, 0] += (int(y) - self.maxR)
        circI[:, 1] += (int(x) - self.maxR)

        #odstraneni bodu mimo obraz
        circI = circI[circI[:, 1] >= 0, :]  # x coord
        circI = circI[circI[:, 1] < self.mask[self.currFrameIdx, :, :].shape[1], :]  # x coord
        circI = circI[circI[:, 0] >= 0, :]  # y coord
        circI = circI[circI[:, 0] < self.mask[self.currFrameIdx, :, :].shape[0], :]  # y coord

        if self.mask[self.currFrameIdx, :, :].any():
            self.mask[self.currFrameIdx, circI[:, 0], circI[:, 1]] = 0
        self.seeds[self.currFrameIdx, circI[:, 0], circI[:, 1]] = 0

        self.memskel.skel[self.currFrameIdx, :, :] = np.zeros_like(self.memskel.skel[self.currFrameIdx, :, :])
        self.redrawFig()

    def getMarkers(self):
        return self.seeds[self.currFrameIdx, :, ]

    def setInitMask( self, mask, idx=-1, redraw=True, setSeeds=True, setMask=True ):
        if idx == -1:
            idx = self.currFrameIdx
        if setSeeds:
            self.seeds[idx,:,:] = mask
        if setMask:
            self.mask[idx,:,:] = mask
        self.linesL = list()
        if redraw:
            self.redrawFig()

    def resetFrame(self, idx=-1):
        if idx == -1:
            idx = self.currFrameIdx
        self.seeds[idx,:,:] = np.zeros_like( self.seeds[idx,:,:] )
        self.mask[idx,:,:] = np.zeros_like( self.mask[idx,:,:] )
        self.memskel.skel[idx,:,:] = np.zeros_like( self.memskel.skel[idx,:,:] )
        self.redrawFig()

    def registerCallbacks(self, mode):
        if mode == 'mark':
            self.memskel.canvas.get_tk_widget().config(cursor='target')
            self.changeMade = False
            self.mousepressCID  = self.fig.canvas.mpl_connect('button_press_event', self.on_mousePress)
            self.mousereleaseCID = self.fig.canvas.mpl_connect('button_release_event', self.on_mouseRelease)
            self.mousemoveCID = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouseMove)
            self.mousescrollCID = self.fig.canvas.mpl_connect('scroll_event', self.on_mouseScrollMarkMode)
            self.keypressCID = self.fig.canvas.mpl_connect('key_press_event', self.on_keypressM)
        elif mode == 'view':
            self.memskel.canvas.get_tk_widget().config(cursor='arrow' )
            self.mousepressCID  = self.fig.canvas.mpl_connect('button_press_event', self.on_mousePress)
            self.mousereleaseCID = self.fig.canvas.mpl_connect('button_release_event', self.on_mouseRelease)
            self.mousemoveCID = self.fig.canvas.mpl_connect('motion_notify_event', self.on_mouseMove)
            self.mousescrollCID = self.fig.canvas.mpl_connect('scroll_event', self.on_mouseScrollViewMode)
            self.keypressCID = self.fig.canvas.mpl_connect('key_press_event', self.on_keypressM)
        else:
            warnings.warn('Wrong mode passed.')

    def unregisterCallbacks(self, mode):
        if mode == 'mark':
            self.fig.canvas.mpl_disconnect(self.mousepressCID)
            self.fig.canvas.mpl_disconnect(self.mousereleaseCID)
            self.fig.canvas.mpl_disconnect(self.mousemoveCID)
            self.fig.canvas.mpl_disconnect(self.mousescrollCID)
            self.fig.canvas.mpl_disconnect(self.keypressCID)
            self.memskel.canvas.get_tk_widget().config(cursor='arrow')
        elif mode == 'view':
            self.fig.canvas.mpl_disconnect(self.mousepressCID)
            self.fig.canvas.mpl_disconnect(self.mousereleaseCID)
            self.fig.canvas.mpl_disconnect(self.mousemoveCID)
            self.fig.canvas.mpl_disconnect(self.mousescrollCID)
            self.fig.canvas.mpl_disconnect(self.keypressCID)
            self.memskel.canvas.get_tk_widget().config(cursor='arrow')
    #        cv2.setMouseCallback( self.windowname, lambda event, x, y, flags, param:None )
        else:
            warnings.warn('Wrong mode passed.')

    def drawContours(self):
        plt.contour(self.xgrid, self.ygrid, self.mask[self.currFrameIdx, :, :], levels=[0], colors='r')
        plt.draw()

    def redrawFig(self):
        plt.hold(False)
        plt.imshow(self.img, aspect='equal')
        plt.hold(True)

        for i in self.linesL:
            self.ax.add_artist(i)
        # self.ax.add_artist( self.text )
        if self.isErasing:
            self.ax.add_artist(self.eraser)
        if self.memskel.dispThresh.get() and self.memskel.maskThresh[self.currFrameIdx, :, :].any():
            # mask_cmap = matcol.colorConverter.to_rgba('blue')
            # mask_cmap[:, :, -1] = self.memskel.maskThresh
            plt.imshow(self.memskel.maskThresh[self.currFrameIdx, :, :], cmap='jet', alpha=0.3, interpolation='nearest')
        if self.memskel.dispMem.get() and self.mask[self.currFrameIdx, :, :].any():
            self.drawContours()
        if self.memskel.dispSpline.get() and self.memskel.approxSkel[self.currFrameIdx].any(): #in case approximation was done
            # labels = mt.fuseImages( (self.memskel.approxSkel[self.currFrameIdx,:,:], self.memskel.approxSkel[self.currFrameIdx,:,:]), 'bb' )
            plt.plot(self.memskel.approxSkel[self.currFrameIdx][1], self.memskel.approxSkel[self.currFrameIdx][0], 'b', linewidth=3, scalex=False, scaley=False)
            plt.plot(self.memskel.approxSkel[self.currFrameIdx][1], self.memskel.approxSkel[self.currFrameIdx][0], 'm', linewidth=2, scalex=False, scaley=False)
            # plt.imshow( labels )
        if self.memskel.dispSkel.get() and self.memskel.skel[self.currFrameIdx, :, :].any():
            labels = mt.fuseImages((self.memskel.skel[self.currFrameIdx, :, :], self.memskel.skel[self.currFrameIdx, :, :]), 'bb')
            plt.imshow(labels)

        # if not self.memskel.dispMem.get() and not self.memskel.dispSpline.get() and not self.memskel.dispSkel.get():
        #     plt.hold( False )
        #
        #     plt.imshow( np.zeros_like(self.img) )
        #     plt.hold( True )
        #     #plt.imshow( self.mask[self.currFrameIdx,:,:] )
        #     plt.plot(self.memskel.approxSkel[self.currFrameIdx][1], self.memskel.approxSkel[self.currFrameIdx][0], 'w', linewidth=64, scalex=False, scaley=False)
        #     plt.plot(self.memskel.approxSkel[self.currFrameIdx][1], self.memskel.approxSkel[self.currFrameIdx][0], 'm', linewidth=25, scalex=False, scaley=False)

        self.fig.canvas.draw()
        #        plt.imshow( self.dests[0], aspect='equal' )

        #    def update(self):
        #        plt.hold( False )
        #        plt.imshow( self.im, aspect = 'equal', extent = None )
        #        plt.hold(True)
        #        if self.initType == 'circle':
        #            self.artist = Circle( (self.y, self.x), self.rad, color = 'r', fill = False, linewidth = 20 )
        #            self.ax.add_artist( self.artist )
        #        elif self.initType == 'seeds':
        #            layer = mt.imageLayer( self.seeds, 'r')
        #            plt.imshow( layer )
        #        #        plt.hold(True)
        #        plt.draw()

        #    def setContours( self, contours ):
        #        if contours.shape == self.dests[1].shape:
        #            self.contours = contours
        #        else:
        #            self.contours = cv2.resize( contours.astype(np.uint8), self.visImg[1].shape, interpolation=cv2.INTER_NEAREST )