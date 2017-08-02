import numpy as np
import pymorph as pm
from scipy import interpolate


def sequential_thinning(skel, type='both'):
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


def skel2path(skel, startPt=[]):
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


def approximate(data, smoothing_fac):
    pathsL = []

    # nProcessedFrames = np.amax(np.amax(data, axis=1), axis=1).sum()
    m = data.sum()# / nProcessedFrames  # average number of points in skelet
    maxsf = m + np.sqrt(2 * m)  # maximal recommended smoothing factor according to scipy documentation
    sfLevel = int(smoothing_fac)
    nLevels = 10  # number of smoothing levels
    sf = int(sfLevel * ((2 * maxsf) / nLevels))

    path = skel2path(data.copy())
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

    # self.data.spline[idx] = (tck, u)
    # self.data.approx_skel[idx] = np.array(interpolate.splev(x=u, tck=tck))
    approx = np.array(interpolate.splev(x=u, tck=tck))
    return (tck, u), approx