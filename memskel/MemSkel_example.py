__author__ = 'Ryba'
import MemSkel as ms

ms.MemSkel().run()

#-------------------------------------------------------
#import matplotlib.pyplot as plt
#import numpy as np
#import cv2
#from scipy import interpolate
#import cv2
#
#filename = 'z:/Work/Bunky/images/skel/skel_003.png'
#
#img = cv2.imread(filename,0)
#
#skel = img.copy()
#
#nPoints = len( np.nonzero( skel )[0] )
#path = np.zeros( (nPoints, 2) )
#
#startPt = np.argwhere( skel )[0,:]
#skel[ startPt[0], startPt[1] ] = 0
#path[0,:] = startPt
#
#nghbsI = np.array( ([-1,0],[0,-1],[0,1],[1,0], [-1,-1],[-1,1],[1,-1],[1,1]) )
#
##while skel.any():
#currPt = startPt
#for i in range( 1, nPoints - 1 ):
#    mask = np.array( [ nghbsI[:,0] + currPt[0] , nghbsI[:,1] + currPt[1] ] ).conj().transpose()
#    nghbs = skel[ mask[:,0], mask[:,1] ]
#    firstI = np.argwhere( nghbs )[0] #get index of first founded neighbor
#    currPt = np.squeeze( mask[ firstI, : ] ) #get that neighbor
#    path[i,:] = currPt
#    skel[ currPt[0], currPt[1] ] = 0
#
#path[-1,:] = np.argwhere( skel )
#
#skelPts = path
#x = []
#y = []
#for point in skelPts:
#    x.append(point[0])
#    y.append(point[1])
#
#addn = 3 + 1
#for i in range(addn):
#    x.append( x[i] )
#    y.append( y[i] )
#
##---------------------
#
#m = len(x) #number of points
#mins = m - np.sqrt(2*m)
#maxs = m + np.sqrt(2*m)
#
#unew = np.arange(0,1.001,0.001)
#
#plt.figure(), plt.gray()
#i = 0
#for sc in range(0, int(2*maxs) , int(2*maxs/4)):
#    i += 1
#    tck, u = interpolate.splprep( [x,y], s = sc, per=1 )
#    out = np.array( interpolate.splev(x = u, tck = tck) )
#    plt.subplot(2,3,i), plt.imshow(np.zeros_like(img)), plt.hold(True), plt.plot(out[1], out[0], 'r'), plt.axis('image'), plt.title('s= ' + str(sc) + ', knots#=' + str(len(tck[0])))
#
#plt.show()