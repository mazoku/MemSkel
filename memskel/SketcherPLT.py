#import cv2
import matplotlib.pyplot as plt
import matplotlib.lines as lines
#import matplotlib.text as text
import pymorph as pm
from pylab import *
import numpy as np
import cv2
import cv2.cv as cv
import warnings

class SketcherPLT:
    def __init__(self, windowname, dests, colors_func, on_keypress ):
        self.w, self.h = figaspect( dests[1] )
        self.fig = plt.figure( figsize=(self.w,self.h) )
        plt.gray()
        plt.hold( True )
#        self.ax = self.fig.add_subplot(111) #self.fig.add_axes( (0,0,1,1) )
#        plt.imshow( dests[0] )
        self.ax = Axes( self.fig, [0,0,1,1], yticks=[], xticks=[], frame_on=False )
        self.fig.delaxes( plt.gca() )
        self.fig.add_axes( self.ax )
        plt.imshow( dests[0], aspect='equal' )

        self.dests = dests
        self.colors_func = colors_func
        self.lineWidth = 5
        self.text = Text( x = 10, y = 10, text = 'linewidth = ' + str(self.lineWidth), color = 'green' )
        self.ax.add_artist( self.text )
        self.linesL = list()
        self.line = None
        self.lineMarkers = None
#        self.contours = None
        self.changeMade = False
        self.eraser = None
        self.eraserR = 5
        self.maxR = 20
        self.minR = 3
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

        self.mask = np.zeros( self.dests[1].shape, dtype=np.bool )


    def run(self):
        self.registerCallbacks()
        plt.show()


    def on_mousePress( self, event ):
        if event.inaxes and event.button == 1:
            self.isMarking = True
            self.lineMarkers = np.zeros( self.dests[1].shape, dtype=np.int32 )
            self.line = Line2D( [round(event.xdata)], [round(event.ydata)], linewidth=self.lineWidth, color='red' )
            self.linesL.append( self.line )
            self.ax.add_artist( self.line )
            plt.draw()
        elif event.inaxes and event.button == 3:
            self.eraser = Circle( (event.xdata, event.ydata), self.lineWidth, color = 'b', fill = False, linewidth = self.eraserR )
            self.ax.add_artist( self.eraser )
            self.isErasing = True
            self.text.set_text( 'eraser radius = ' + str(self.eraserR) )
            self.erase( event.xdata, event.ydata )
            plt.draw()


    def on_mouseRelease( self, event ):
        if event.button == 1:
            self.isMarking = False
            self.lineMarkers = pm.dilate( self.lineMarkers.astype(np.bool) , np.ones( (self.lineWidth,self.lineWidth), dtype=np.bool ) )
            self.dests[1] |= np.copy( self.lineMarkers )
        elif event.button == 3:
            self.isErasing = False
            self.eraser.remove()
            plt.draw()


    def on_mouseMove( self, event):
        if event.inaxes and self.isMarking:
#            pt = ( round(event.ydata), round(event.xdata) )
            if event.button == 1:
                xd = np.hstack( (self.line.get_xdata(), event.xdata) )
                yd = np.hstack( (self.line.get_ydata(), event.ydata) )
                self.line.set_xdata( xd )
                self.line.set_ydata( yd )
                plt.draw()
                self.lineMarkers[event.ydata, event.xdata] = 1
#                self.dests[1][event.ydata, event.xdata] = 1
                self.changeMade = True
            else:
                self.prev_pt = None
        elif self.isMarking and not event.inaxes:
            self.isMarking = False
            self.lineMarkers = pm.dilate( self.lineMarkers, np.ones( (self.lineWidth,self.lineWidth), dtype=np.int ) )
            self.dests[1] |= self.lineMarkers
        elif self.isErasing:
            self.eraser.remove()
            self.eraser = Circle( (event.xdata, event.ydata), self.eraserR, color = 'b', fill = False, linewidth = 5 )
            self.ax.add_artist( self.eraser )

            self.erase( event.xdata, event.ydata )

            plt.draw()
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
        circ = cv2.distanceTransform( circ, cv.CV_DIST_L2, 3)
        return circ


    def on_mouseScroll( self, event):
        if self.isErasing:
            self.eraserR = int( self.eraserR + event.step )
            if event.step > 0:
                self.eraserR = min( self.eraserR, self.maxR )
            else:
                self.eraserR = max( self.eraserR, self.minR )
            self.eraser.remove()
            self.eraser = Circle( (event.xdata, event.ydata), self.eraserR, color = 'b', fill = False, linewidth = 5)
            self.ax.add_artist( self.eraser )
            self.text.set_text( 'eraser radius = ' + str(self.eraserR) )
            plt.draw()
        else:
            maxLW = 20
            self.lineWidth = int( self.lineWidth + event.step )
            if event.step > 0:
                self.lineWidth = min( self.lineWidth, maxLW )
            else:
                self.lineWidth = max( self.lineWidth, 1 )
            self.text.set_text( 'linewidth = ' + str(self.lineWidth) )
            plt.draw()


    def erase( self, x, y ):
        circI = np.argwhere( self.eraserMask <= self.eraserR )
        circI[:,0] += (y - self.maxR)
        circI[:,1] += (x - self.maxR)

        #odstraneni bodu mimo obraz
        circI = circI[circI[:,1] >= 0,:] #x coord
        circI = circI[circI[:,1] < self.mask.shape[1],:] #x coord
        circI = circI[circI[:,0] >= 0,:] #y coord
        circI = circI[circI[:,0] < self.mask.shape[0],:] #y coord

        if self.mask.any():
            self.mask[ circI[:,0], circI[:,1] ] = 0
        self.dests[1][ circI[:,0], circI[:,1] ] = 0

        self.redrawFig()


    def getMarkers( self ):
        return self.dests[1]


    def setInitMask( self, mask ):
        self.dests[1] = mask
        self.linesL = list()
        self.redrawFig()


    def registerCallbacks( self ):
        self.changeMade = False
        self.mousepressCID  = self.fig.canvas.mpl_connect( 'button_press_event', self.on_mousePress )
        self.mousereleaseCID = self.fig.canvas.mpl_connect( 'button_release_event', self.on_mouseRelease )
        self.mousemoveCID = self.fig.canvas.mpl_connect( 'motion_notify_event', self.on_mouseMove )
        self.mousescrollCID = self.fig.canvas.mpl_connect( 'scroll_event', self.on_mouseScroll )
        self.keypressCID = self.fig.canvas.mpl_connect( 'key_press_event', self.on_keypressM )


    def unregisterCallbacks( self ):
        self.fig.canvas.mpl_disconnect( self.mousepressCID )
        self.fig.canvas.mpl_disconnect( self.mousereleaseCID )
        self.fig.canvas.mpl_disconnect( self.mousemoveCID )
        self.fig.canvas.mpl_disconnect( self.mousescrollCID )
        self.fig.canvas.mpl_disconnect( self.keypressCID )
#        cv2.setMouseCallback( self.windowname, lambda event, x, y, flags, param:None )


    def drawContours( self ):
        x = np.arange( 0, self.dests[1].shape[1] )
        y = np.arange( 0, self.dests[1].shape[0] )
        xgrid, ygrid = np.meshgrid( x, y )
        plt.contour( xgrid, ygrid, self.mask, levels = [0], colors = 'r')
        plt.draw()


    def redrawFig( self ):
        plt.hold( False )
        plt.imshow( self.dests[0], aspect='equal' )
        plt.hold( True )

        for i in self.linesL:
            self.ax.add_artist( i )
        self.ax.add_artist( self.text )
        if self.isErasing:
            self.ax.add_artist( self.eraser )
        if self.mask.any():
            self.drawContours( )
        plt.draw()
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