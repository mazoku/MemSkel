__author__ = 'Ryba'

import numpy as np
import pymorph as pm
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def sequentialThinning(skel, type = 'both'):
    golayL1Hit = np.array( [[0,0,0], [0,1,0], [1,1,1]] )
    golayL1Miss = np.array( [[1,1,1], [0,0,0], [0,0,0]] )
    golayL = pm.se2hmt( golayL1Hit, golayL1Miss )

    golayE1Hit = np.array( [[0,1,0], [0,1,0], [0,0,0]] )
    golayE1Miss = np.array( [[0,0,0], [1,0,1], [1,1,1]] )
    golayE = pm.se2hmt( golayE1Hit, golayE1Miss )

    if type == 'golayL' or type == 'both':
        skel = pm.thin(skel, Iab = golayL, n = -1, theta = 45, direction = 'clockwise')
    if type == 'golayE' or type == 'both':
        skel = pm.thin(skel, Iab = golayE, n = -1, theta = 45, direction = 'clockwise')

    #    changed = True
    #    while changed:
    #        skelT = pm.thin(skel, Iab = golayE, n = 1, theta = 45, direction = "clockwise")
    #        plt.figure()
    #        plt.subplot(121), plt.imshow( skel ), plt.gray()
    #        plt.subplot(122), plt.imshow( skelT ), plt.gray()
    #        plt.show()
    #
    #        if (skel == skelT).all() :
    #            changed = False
    #        skel = skelT

    return skel


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def fuseImages(imgs, colors, type='imgs', shape=[]):
    """
    Function for fusing multiple binary images.
    Supported colors are:
        w ... white
        r ... red
        g ... green
        b ... blue
        y ... yellow
        m ... magenta
        c ... cyan
    Supported types are:
        imgs ... images are given in matrix form
        coords ... images are given by non-zero pixels in tuple form
    """
    colorsD = dict({'w' : [1,1,1,1],
                    'r' : [1,0,0,1],
                    'g' : [0,1,0,1],
                    'b' : [0,0,1,1],
                    'y' : [1,1,0,1],
                    'm' : [1,0,1,1],
                    'c' : [0,1,1,1],
                    'x' : [0,0,0,0]})

    if shape:
        finalIm = np.zeros(np.hstack((shape, 4)))
        finalIm[:, :, 3] = 1
    else:
        finalIm = np.zeros(np.hstack((imgs[0].shape, 4)))

    for i in range(len(imgs)):
        im = imgs[i]
        if not im.any():
            continue
        c = colors[i]
        if type == 'imgs':
            imP = np.argwhere(im)
        elif type == 'coords':
            imP = im.astype(np.int)
        if len(imP.shape) > 1:
            finalIm[imP[:, 0], imP[:, 1], :] = colorsD[c]
        else:
            finalIm[imP[0], imP[1], :] = colorsD[c]

    return finalIm



#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def imageLayer( img, color, type = 'imgs', shape = [] ):
    """

    """
    colorsD = dict({ 'w' : [1,1,1,1],
                     'r' : [1,0,0,1],
                     'g' : [0,1,0,1],
                     'b' : [0,0,1,1],
                     'y' : [1,1,0,1],
                     'm' : [1,0,1,1],
                     'c' : [0,1,1,1],
                     'x' : [0,0,0,0]})

    if shape:
        finalIm = np.zeros( np.hstack( (shape, 4) ) )

    else:
        finalIm = np.zeros( np.hstack( (img.shape, 4) ) )

    if not img.any():
        return finalIm
    if type == 'imgs':
        imP = np.argwhere( img )
    elif type == 'coords':
        imP = img
    if len(imP.shape) > 1:
        finalIm[ imP[:,0], imP[:,1], : ] = colorsD[ color ]
    else:
        finalIm[ imP[0], imP[1], : ] = colorsD[ color ]

    return finalIm



#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def skel2path( skel, startPt = [], disp = False ):
    nPoints = len( np.nonzero( skel )[0] )
    path = np.zeros( (nPoints, 2) )

    if not startPt:
        startPt = np.argwhere( skel )[0,:]
    skel[ startPt[0], startPt[1] ] = 0
    path[0,:] = startPt

    nghbsI = np.array( ([-1,0],[0,-1],[0,1],[1,0], [-1,-1],[-1,1],[1,-1],[1,1]) )

    #while skel.any():
    currPt = startPt
    for i in range( 1, nPoints - 1 ):
        mask = np.array( [ nghbsI[:,0] + currPt[0] , nghbsI[:,1] + currPt[1] ] ).conj().transpose()
        nghbs = skel[ mask[:,0], mask[:,1] ]
        firstI = np.argwhere( nghbs )[0] #get index of first founded neighbor
        currPt = np.squeeze( mask[ firstI, : ] ) #get that neighbor
        path[i,:] = currPt
        skel[ currPt[0], currPt[1] ] = 0

        if disp:
            plt.figure(), plt.gray()
            labels = fuseImages( ( np.argwhere( skel ), path, currPt), shape = skel.shape, colors = 'wrb', type = 'coords' )
            plt.imshow( labels )
            plt.show()
    path[-1,:] = np.argwhere( skel ) #should be the last remaining pixel

    return path



#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def absdiffNP( a1, a2 ):
    diff = np.absolute( a1.astype( np.float ) - a2.astype( np.float ) ).astype( a1.dtype )
    return diff