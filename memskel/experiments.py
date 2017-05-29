from __future__ import division

import numpy as np
from numpy.lib.stride_tricks import as_strided as ast
import matplotlib.pyplot as plt
import cv2

import skimage.io as skiio
import skimage.color as skicol
import skimage.filters as skifil
import skimage.util as skiuti
import skimage.morphology as skimor
import skimage.segmentation as skiseg
import skimage.exposure as skiexp
import skimage.measure as skimea

from itertools import product

# import skfuzzy
from PIL import Image


def norm_shape(shape):
    '''
    Normalize numpy array shapes so they're always expressed as a tuple,
    even for one-dimensional shapes.

    Parameters
        shape - an int, or a tuple of ints

    Returns
        a shape tuple
    '''
    try:
        i = int(shape)
        return (i,)
    except TypeError:
        # shape was not a number
        pass

    try:
        t = tuple(shape)
        return t
    except TypeError:
        # shape was not iterable
        pass

    raise TypeError('shape must be an int, or a tuple of ints')


def sliding_window(a, ws, ss=None, flatten=True):
    '''
    Return a sliding window over a in any number of dimensions

    Parameters:
        a  - an n-dimensional numpy array
        ws - an int (a is 1D) or tuple (a is 2D or greater) representing the size
             of each dimension of the window
        ss - an int (a is 1D) or tuple (a is 2D or greater) representing the
             amount to slide the window in each dimension. If not specified, it
             defaults to ws.
        flatten - if True, all slices are flattened, otherwise, there is an
                  extra dimension for each dimension of the input.

    Returns
        an array containing each n-dimensional window from a
    '''

    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)

    # convert ws, ss, and a.shape to numpy arrays so that we can do math in every
    # dimension at once.
    ws = np.array(ws)
    ss = np.array(ss)
    shape = np.array(a.shape)


    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shape),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shape):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))

    # how many slices will there be in each dimension?
    newshape = norm_shape(((shape - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    strided = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return strided

    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    dim = filter(lambda i : i != 1,dim)
    return strided.reshape(dim)


def milousovo(im, winw=51, step=1, rad=3, show=False):
    im_orig = im.copy()

    # im_op = skimor.opening(im, selem=skimor.disk(1))
    # im_op = skimor.closing(im, selem=skimor.disk(1))

    # plt.figure()
    # plt.subplot(121), plt.imshow(im_orig, 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(im, 'gray', interpolation='nearest')
    # plt.show()

    # print 'Sliding ...',
    # sw = sliding_window(im, (winw, winw), (step, step))
    # print 'done'
    #
    # # af = np.zeros((im.shape[0] - (winw - 1), im.shape[1] - (winw - 1)))
    # # af = im[winw//2:im.shape[0] - winw//2, winw//2:im.shape[1] - winw//2]
    # af = skiuti.crop(im, ((winw//2, winw//2), (winw//2, winw//2)), copy=True)
    # rows, cols = np.unravel_index(range(sw.shape[0]), af.shape)
    # for i, w in enumerate(sw):
    #     af[rows[i], cols[i]] -= w.min()
    # # af_op = skimor.opening(af, selem=skimor.disk(rad))
    # af_op = skimor.closing(af, selem=skimor.disk(rad))

    # --------------------------------------------------
    # im_mor = skimor.opening(im_orig, selem=skimor.disk(rad))
    im_mor = skimor.closing(im_orig, selem=skimor.disk(rad))
    # print 'Sliding ...',
    # sw = sliding_window(im_mor, (winw, winw), (step, step))
    # print 'done'
    #
    # # af = np.zeros((im.shape[0] - (winw - 1), im.shape[1] - (winw - 1)))
    # # af = im[winw//2:im.shape[0] - winw//2, winw//2:im.shape[1] - winw//2]
    # af = skiuti.crop(im_mor, ((winw//2, winw//2), (winw//2, winw//2)), copy=True)
    # rows, cols = np.unravel_index(range(sw.shape[0]), af.shape)
    # for i, w in enumerate(sw):
    #     af[rows[i], cols[i]] -= w.min()
    af = skifil.median(im_mor, selem=skimor.disk(3))

    if show:
        plt.figure()
        plt.subplot(131), plt.imshow(im_orig, 'gray', interpolation='nearest'), plt.title('input image')
        plt.subplot(132), plt.imshow(im_mor, 'gray', interpolation='nearest'), plt.title('opening')
        plt.subplot(133), plt.imshow(af, 'gray', interpolation='nearest'), plt.title('milous')

        # plt.subplot(234), plt.imshow(im_orig, 'gray', interpolation='nearest'), plt.title('input image')
        # plt.subplot(235), plt.imshow(im_op, 'gray', interpolation='nearest'), plt.title('opening')
        # plt.subplot(236), plt.imshow(af2, 'gray', interpolation='nearest'), plt.title('milous')
        plt.show()

    #TODO: best res - closing -> milous
    #TODO: misto MIN zkusit MEAN nebo MEDIAN

    return af

if __name__ == '__main__':
    fname = '/home/tomas/Dropbox/Data/MemSkel/c11_Gi1(N269D)-L91-YFP(wurz)+Gb1+Gy2_392_10us_960nm_50mW_unidir_75-150_740DCXR_542-27_up_700.tif'

    im = skiio.imread(fname)
    img = skicol.rgb2gray(im.swapaxes(0, 1).swapaxes(1, 2))

    # percent thresh
    hist, bins = skiexp.histogram(img)
    hist = hist / hist.sum()
    cum_hist = hist.copy()
    for i in range(1, len(cum_hist)):
        cum_hist[i] += cum_hist[i - 1]

    perc_ts = [0.6, 0.7, 0.75, 0.8, 0.85, 0.9]
    colors = 5 * 'rgbcmyk'
    colors = colors[:len(perc_ts)]
    thresholds = []
    threshold_inds = []
    im_percs = []
    for p in perc_ts:
        diff = cum_hist - p
        diff *= diff > 0

        t_ind = np.nonzero(diff)[0][0]
        t = bins[t_ind]
        threshold_inds.append(t_ind)
        thresholds.append(t)

        res = img > t
        im_percs.append(res)

    plt.figure()
    plt.subplot(211), plt.plot(bins, hist, 'b-')
    plt.subplot(212), plt.plot(bins, cum_hist, 'r-')
    for t, t_ind, c in zip(thresholds, threshold_inds, colors):
        plt.plot((t, t), (0, cum_hist[t_ind]), '%s-' % c)
        plt.plot((0, t), (cum_hist[t_ind], cum_hist[t_ind]), '%s-' % c)
        plt.plot(t, cum_hist[t_ind], '%so' % c)

    plt.figure()
    n_imgs = len(im_percs)
    for i, (res, t, p_t) in enumerate(zip(im_percs, thresholds, perc_ts)):
        plt.subplot(1, n_imgs, i + 1)
        plt.imshow(res, 'gray', interpolation='nearest')
        plt.title('p=%.2f, t=%.3f' % (p_t, t))
    plt.show()

    # OTSU
    print 'Otsu ...',
    thresh_otsu = skifil.threshold_otsu(img)
    im_otsu = img > thresh_otsu
    print 'done'

    # ADAPTIVE
    # print 'Adaptive ...',
    # block_size = 100
    # im_adaptive = skifil.threshold_adaptive(img, block_size)
    # print 'done'

    # Milousovo
    print 'Milousovo ...',
    mil = milousovo(img, winw=15, step=1, show=False)
    print 'done'

    # # LI
    # print 'Li ...',
    # thresh_li = skifil.threshold_li(img)
    # im_li = img > thresh_li
    # print 'done'

    # # YEN
    # print 'Yen ...',
    # nbins = 256
    # thresh_yen = skifil.threshold_yen(img, nbins=nbins)
    # im_yen = img > thresh_yen
    # print 'done'

    # # ISODATA
    # print 'Isodata ...',
    # nbins = 256
    # thresh_iso = skifil.threshold_isodata(img, nbins=nbins)
    # im_iso = img > thresh_iso
    # print 'done'

    # Fuzzy c-means
    # print 'Fuzzy CMeans ...',
    # n_clusters = 2
    # data = img.flatten().reshape((1, np.prod(img.shape)))
    # cntr, u, u0, d, jm, p, fpc = skfuzzy.cluster.cmeans(data, n_clusters, m=2, error=0.005, maxiter=1000)
    # cm = 1 - np.argmax(u, axis=0)  # cluster membership
    # cm = cm.reshape(img.shape)
    # print 'done'

    # Morphology
    im_otsu = skimor.binary_opening(im_otsu, selem=skimor.disk(1))
    im_otsu = skimor.binary_closing(im_otsu, selem=skimor.disk(3))

    block_size = 100
    mil_adapt = skifil.threshold_adaptive(mil, block_size)

    mil_adapt = skimor.binary_opening(mil_adapt, selem=skimor.disk(3))
    mil_adapt = skimor.binary_closing(mil_adapt, selem=skimor.disk(5))

    # plt.figure()
    # plt.gray()
    # plt.subplot(231), plt.imshow(img), plt.title('Image'), plt.axis('off')
    # plt.subplot(232), plt.imshow(im_otsu), plt.title('Otsu'), plt.axis('off')
    # # plt.subplot(233), plt.imshow(im_adaptive), plt.title('Adaptive'), plt.axis('off')
    # plt.subplot(234), plt.imshow(mil), plt.title('Milous'), plt.axis('off')
    # plt.subplot(235), plt.imshow(mil_adapt), plt.title('Milous adapt'), plt.axis('off')
    # plt.subplot(334), plt.imshow(im_li), plt.title('Li'), plt.axis('off')
    # plt.subplot(335), plt.imshow(im_yen), plt.title('Yen'), plt.axis('off')
    # plt.subplot(336), plt.imshow(im_iso), plt.title('Isodata'), plt.axis('off')
    # plt.subplot(337), plt.imshow(cm), plt.title('FCM'), plt.axis('off')

    # plt.figure()
    # plt.imshow(skiseg.mark_boundaries(img, im_otsu.astype(np.int8)), 'gray', interpolation='nearest'), plt.title('boundaries')
    # plt.show()

    # img_int = skiuti.img_as_uint(img)
    # cnts_otsu = skimea.find_contours(im_otsu, 0.9)
    # plt.figure()
    # plt.imshow(img_int, 'gray', interpolation='nearest')
    # plt.hold(True)
    # for c in cnts:
    #     plt.plot(c[:, 1], c[:, 0], 'y',linewidth=2)
    #     plt.axis('image')
    # plt.show()

    # hledani nejvetsi komponenty
    labels = skimea.label(im_otsu, connectivity=2, background=0)
    areas = [(labels == x).sum() for x in np.unique(labels) if x > -1]
    otsu_max_lab = labels == np.argmax(areas)

    labels = skimea.label(mil_adapt, connectivity=2, background=0)
    areas = [(labels == x).sum() for x in np.unique(labels) if x > -1]
    af_max_lab = labels == np.argmax(areas)

    plt.figure()
    plt.subplot(221), plt.imshow(img, 'gray', interpolation='nearest'), plt.title('input')
    plt.subplot(222), plt.imshow(otsu_max_lab, 'gray', interpolation='nearest'), plt.title('otsu mask')
    plt.subplot(223), plt.imshow(img * otsu_max_lab, 'gray', interpolation='nearest'), plt.title('masked')
    # plt.subplot(223), plt.imshow(img * im_otsu, 'gray', interpolation='nearest'), plt.title('masked')
    # plt.subplot(133), plt.imshow(img, 'gray', interpolation='nearest')
    # cnts_otsu = skimea.find_contours(im_otsu, 0.9)
    # plt.hold(True)
    # for c in cnts_otsu:
    #     plt.plot(c[:, 1], c[:, 0], 'y',linewidth=1)
    #     plt.axis('image')

    plt.figure()
    plt.subplot(221), plt.imshow(img, 'gray', interpolation='nearest'), plt.title('input')
    plt.subplot(222), plt.imshow(af_max_lab, 'gray', interpolation='nearest'), plt.title('adaptive filtering mask')
    plt.subplot(223), plt.imshow(img * af_max_lab, 'gray', interpolation='nearest'), plt.title('masked')
    # plt.subplot(133), plt.imshow(img, 'gray', interpolation='nearest')
    # cnts_af = skimea.find_contours(mil_adapt, 0.9)
    # plt.hold(True)
    # for c in cnts_af:
    #     plt.plot(c[:, 1], c[:, 0], 'y',linewidth=1)
    #     plt.axis('image')

    plt.show()
