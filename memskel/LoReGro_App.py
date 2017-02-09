import numpy as np
import matplotlib.pyplot as plt
import cv2
import Sketcher
import SketcherPLT
import pymorph as pm
import myTools as mt
from pylab import *
import os
from skimage.io import MultiImage
import string
import time

help_message = '''
  USAGE: watershed.py [<image>]

  Use keys 1 - 7 to switch marker color
  SPACE - update segmentation
  r     - reset
  a     - switch autoupdate
  ESC   - exit

'''

class LoReGro_App:
    def __init__( self, fn, energy = np.empty(0), scaleFactor = 4 ):
        if isinstance(fn, str):
            file, ext = os.path.splitext(fn)
            if ext == '.tif' or ext == '.tiff':
                self.imMI  = MultiImage( fn )
                self.numframes = len( self.imMI )
                self.imIdx = 0
                self.img = self.imMI[self.imIdx]
            else:
                self.img = cv2.imread(fn)
        elif isinstance(fn, np.ndarray ):
            self.img = fn
        else:
            print( 'Unknown image format' )
            return
        h, w = self.img.shape[:2]
        if energy.any():
            self.energy = energy
        else:
            if len( self.img.shape ) == 3:
                self.energy = cv2.cvtColor( self.img, cv2.COLOR_BGR2GRAY )
            else:
                self.energy = self.img
        self.markers = np.zeros((h, w), np.int32)
        self.markers_vis = self.img.copy()
        self.cur_marker = 1
        self.scaleFactor = scaleFactor
        self.size = self.img.shape[:2]
#        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255
        self.colors = np.int32( list( ((0,0,0), (0,0,1), (1,0,0)) ) ) * 255
        self.membraneMask = []
        self.newbies = None #nove pridane pixely
        self.auto_update = True
#        self.updtFig = plt.figure()
        self.sketch = SketcherPLT.SketcherPLT( 'segmentation GUI', [self.markers_vis, self.markers], self.get_colors, self.on_keypress )
#        self.keypressCID = self.sketch.fig.canvas.mpl_connect( 'key_press_event', self.on_keypress )
    #        cv2.imshow( 'energy', self.energy )
        self.timeStart = None
        self.timeStack = None
        self.timeEnd = None


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def run( self ):
        self.timeStart = time.time()
        self.sketch.run()

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def on_keypress( self, event ):
        if event.key == 'enter':
            if self.sketch.changeMade:
                    self.newbies = self.sketch.dests[1] > 0
                    self.sketch.unregisterCallbacks()
                    self.run_segmentation(display=True)
#                    self.sketch.setInitMask( self.sketch.getMarkers() | pm.erode( self.membraneMask.astype(np.bool), np.ones((3,3),dtype=np.bool) ) )
                    self.sketch.setInitMask( pm.erode( self.sketch.mask, np.ones((3,3),dtype=np.bool) ) )
                    self.sketch.registerCallbacks()
                    self.sketch.changeMade = False
            else:
                self.skel = self.getSkel( show=True, idx=0 )
                cv2.imwrite( 'z:/Work/Bunky/images/skel/skel_000.png', 255*self.skel )

                if self.numframes > 1:
                    self.timeStack = time.time()
                    self.sketch.unregisterCallbacks()
                    self.segmentStack()

                    self.timeEnd = time.time()
                    s1 = str(self.timeStack-self.timeStart)
                    s2 = str(self.timeEnd-self.timeStack)
                    print 'first frame = ' + s1[:string.rfind(s1,'.')+3] + 's, stack = ' + s2[:string.rfind(s2,'.')+3] + 's\n'


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def segmentStack( self ):
        t = 5
        for i in range(1,self.numframes):
            print 'frame ' + str(i) + ' / ' + str(self.numframes) + '...'
            oldskel = self.skel
            self.imIdx = i
            self.img = self.imMI[self.imIdx]
            self.energy = self.img
            diffim = cv2.absdiff( self.img, self.imMI[self.imIdx-1] )
            mask = (diffim < t )* self.skel
            self.sketch.setInitMask( mask )
            self.newbies = mask
            self.run_segmentation(display=False)
            self.skel = self.getSkel( show=False, idx=i )

            if not self.skel.any() or self.skel.sum() < 10:
                print 'starting over...'
                self.sketch.setInitMask( oldskel )
                self.run_segmentation(display=False)
                self.skel = self.getSkel( show=False, idx=i )

            fn = 'z:/Work/Bunky/images/skel/skel_' + string.zfill(i,3)  + '.png'
            cv2.imwrite( fn, 255*self.skel )
            print 'done\n'
        return 'bla'


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def get_colors( self ):
        return map(int, self.colors[self.cur_marker]), self.cur_marker


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def getSkel( self, show = False, idx = 0 ):
        # extracting skelet of membrane
        memskel = pm.thin( self.membraneMask )
        # processing skelet - removing not-closed parts
        memskel = mt.sequentialThinning( memskel, type = 'golayE' )

        if show:
            # display skelet
            labels = mt.fuseImages( (self.membraneMask, pm.thin( self.membraneMask ) - memskel, memskel), 'wrb' )

            f = plt.figure( figsize=(self.sketch.w,self.sketch.h) )
            f.ax = Axes( f, [0,0,1,1], yticks=[], xticks=[], frame_on=False )
            f.delaxes( plt.gca() )
            f.add_axes( f.ax )
            plt.imshow( self.img, aspect='equal' ), plt.hold( True )
            plt.imshow( labels )
            plt.show()

        #creating image to save
        im = self.img.copy()
        if len(im.shape) == 2:
            im = cv2.cvtColor( im, cv2.COLOR_GRAY2RGB )
        imP = np.argwhere(memskel)
        im[ imP[:,0], imP[:,1], : ] = [255,0,0]
        im = cv2.cvtColor( im, cv2.COLOR_RGB2BGR )

        fn = 'z:/Work/Bunky/images/skel/skelOverlay_' + string.zfill(idx,3)  + '.png'
        cv2.imwrite( fn, im )

        return memskel


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def dataTerm( self, findAll = False ):
        seeds = self.sketch.getMarkers()

        if findAll:
            idxs = np.argwhere( self.markers )
            intensities = np.unique( self.energy[ idxs[:,0],idxs[:,1] ] )
            kernel = [0]
            ints = list()
            for i in range( len(intensities) ):
                for k in kernel:
                    ints.append( intensities[i] + k )
            ints = np.unique( np.array( ints ) )

            for i in ints:
                seeds = np.where( self.energy == i, 1, seeds )

        seeds = seeds > 0

        return seeds


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def run_segmentation( self, display=False ):
        ballROut = 1
        ballR = 3
        minDiff = 0.1
        alphaIB = 1

        mask = self.dataTerm( findAll=False )

        strel = np.ones( (3,3), dtype=np.int )

        ballM = np.zeros( (2*ballR + 1, 2*ballR + 1, 2), dtype = np.int )
        ballM[:,:,1] = np.tile( np.arange(-ballR,ballR+1), [2*ballR + 1,1] )
        ballM[:,:,0] = ballM[:,:,1].conj().transpose()

        ballMOut = np.zeros( (2*ballROut + 1, 2*ballROut + 1, 2), dtype = np.int )
        ballMOut[:,:,1] = np.tile( np.arange(-ballROut,ballROut+1), [2*ballROut + 1,1] )
        ballMOut[:,:,0] = ballMOut[:,:,1].conj().transpose()

        self.membraneMask = self.segmentInnerBoundary( self.energy, mask, ballM, strel, minDiff, alphaIB, display )
        self.membraneMask = pm.close( self.membraneMask )
        self.sketch.mask = self.membraneMask

        self.sketch.redrawFig( )


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def segmentInnerBoundary( self, im, mask, ballM, strel, minDiff, alpha, display=False ):
        itIB = 0
        drawStep = 5
        maxIterations = 1000
        changeMade = True
        while changeMade and itIB < maxIterations:
            itIB += 1
#            mask, accepted, refused = self.iterationIB( im, mask, ballM, strel, minDiff, alpha )
            maskNew, accepted, refused = self.iterationIB2( im, mask.copy(), ballM, strel, minDiff, alpha )
            self.newbies = maskNew - mask
            mask = maskNew
#            self.newbies = np.zeros( self.newbies.shape, dtype=np.bool )
#            self.newbies[accepted[0],accepted[1]] = True
            if display and itIB%drawStep == 0:
                self.sketch.mask = mask
                self.sketch.redrawFig()

            if not accepted.any():
                changeMade = False
        mask = pm.close( mask, np.ones((5,5), dtype=np.int) )
        return mask


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def iterationIB2( self, im, mask, ballM, strel, minDiff, alpha ):
        dilm = pm.dilate( self.newbies, strel )
        newbies2 = dilm - self.newbies
        newbiesL = np.argwhere( newbies2 )#.tolist()

        accepted = list()
        refused = list()

#        labels = mt.fuseImages( (mask, newbies2), 'wg', type = 'imgs')
#        plt.figure()
#        plt.imshow( im ), plt.hold( True )
#        plt.imshow( labels )
#        plt.show()

        maskV = np.argwhere( mask )
        maskV = im[ maskV[:,0], maskV[:,1] ]
        meanMask = maskV.mean()

        for i in range( newbiesL.shape[0] ):
            newbie = newbiesL[ i ]
            inners, outers = self.maskNewbie( newbie, mask, ballM.copy() )

            distIn, distOut = self.getDistances( im, newbie, inners, outers )
            distMask = np.absolute( meanMask - im[ newbie[0], newbie[1] ] )

            weightDistIn = alpha * distIn + (1 - alpha) * distMask

            if weightDistIn < distOut or np.absolute( distIn - distOut ) < minDiff:
                mask[ newbie[0], newbie[1] ] = True
                accepted.append( newbie )
            else:
                refused.append( newbie )

        accepted = np.array( accepted )
        refused = np.array( refused )

        return mask, accepted, refused

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def iterationIB( self, im, mask, ballM, strel, minDiff, alpha ):
        dilm = pm.dilate( mask, strel )
        newbies = dilm - mask
        newbiesL = np.argwhere( newbies )#.tolist()

        accepted = list()
        refused = list()

    #    labels = mt.fuseImages( (mask, newbies), 'wg', type = 'imgs')
    #    plt.figure()
    #    plt.imshow( im ), plt.hold( True )
    #    plt.imshow( labels )
    #    plt.show()

        maskV = np.argwhere( mask )
        maskV = im[ maskV[:,0], maskV[:,1] ]
        meanMask = maskV.mean()

        for i in range( newbiesL.shape[0] ):
            newbie = newbiesL[ i ]
            inners, outers = self.maskNewbie( newbie, mask, ballM.copy() )

            distIn, distOut = self.getDistances( im, newbie, inners, outers )
            distMask = np.absolute( meanMask - im[ newbie[0], newbie[1] ] )

            weightDistIn = alpha * distIn + (1 - alpha) * distMask

            if weightDistIn < distOut or np.absolute( distIn - distOut ) < minDiff:
                mask[ newbie[0], newbie[1] ] = True
                accepted.append( newbie )
            else:
                refused.append( newbie )

        accepted = np.array( accepted )
        refused = np.array( refused )

        return mask, accepted, refused


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def maskNewbie( self, newbie, mask, ballM ):
        ballR = ballM.shape[0] / 2
        ballM[:,:,0] += newbie[0]
        ballM[:,:,1] += newbie[1]

        #    warnings.warn( 'Place for possible speeding up of the process.' )
        #    if np.amin( ballM[:,:,0] ) < 0:
        try:
            while np.amin( ballM[:,:,0] ) < 0:
                ballM = ballM[1:,:,:]
            while np.amin( ballM[:,:,1] ) < 0:
                ballM = ballM[:,1:,:]
            while np.amax( ballM[:,:,0] ) >= mask.shape[0]:
                ballM = ballM[:-1,:,:]
            while np.amax( ballM[:,:,1] ) >= mask.shape[1]:
                ballM = ballM[:,:-1,:]
        except ValueError:
            print 'Error while controling ball boundaries.'

        masked = mask[ ballM[:,:,0], ballM[:,:,1] ]
        innersI = np.nonzero( masked )
        masked[ ballR, ballR ] = True #exclude center from computation
        outersI = np.nonzero( np.invert( masked ) )

        inners = ballM[innersI[0], innersI[1], :]
        outers = ballM[outersI[0], outersI[1], :]

        #    return masked, ballM
        return inners, outers


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
    def getDistances( self, im, newbie, inners, outers ):
        intIn = im[ inners[:,0], inners[:,1] ]
        intOut = im[ outers[:,0], outers[:,1] ]
        intNew = im[ newbie[0], newbie[1] ]

        meanIn = intIn.mean()
        meanOut = intOut.mean()

        distIn = np.absolute( meanIn - intNew )
        distOut = np.absolute( meanOut - intNew )

        return distIn, distOut