from __future__ import division

import matplotlib.pyplot as plt
import numpy as np

import skimage.io as skiio
import skimage.color as skicol
import skimage.exposure as skiexp


def thresh(im, pt=70, t=None):
    '''
    Thresholding - default method is percentile.
    @param im: input image to be thresholded
    @type im: ndarray, dtype=float
    @param pt: percentile threshold, default is 70%
    @type pt: int
    @param t: threshold value, default is None. If it is not None, then this value is used prior to the
    percentile thresholding.
    @type t: float
    @return: thresholded image
    @rtype: ndarray, dtype=booltv
    '''

    if t is None:
        # percent thresh
        hist, bins = skiexp.histogram(img)
        hist = hist / hist.sum()
        cum_hist = hist.copy()
        for i in range(1, len(cum_hist)):
            cum_hist[i] += cum_hist[i - 1]

        diff = cum_hist - pt
        diff *= diff > 0

        t_ind = np.nonzero(diff)[0][0]
        t = bins[t_ind]

    res = img > t
    return res


if __name__ == '__main__':
    fname = '/home/tomas/Dropbox/Data/MemSkel/c11_Gi1(N269D)-L91-YFP(wurz)+Gb1+Gy2_392_10us_960nm_50mW_unidir_75-150_740DCXR_542-27_up_700.tif'

    im = skiio.imread(fname)
    img = skicol.rgb2gray(im.swapaxes(0, 1).swapaxes(1, 2))