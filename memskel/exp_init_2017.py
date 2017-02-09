from __future__ import division

import skimage.io as skiio
import skimage.color as skicol
import skimage.morphology as skimor
import skimage.measure as skimea
import skimage.exposure as skiexp
import skimage.segmentation as skiseg
import skimage.filters as skifil
import skimage.restoration as skires
import skfuzzy as fuzz

import matplotlib.pyplot as plt
import cv2
import numpy as np

from sketcher import Sketcher
from trackbar_param_test import TrackWin

import os
import sys
if os.path.exists('/home/tomas/projects/mrf_segmentation/'):
    # sys.path.append('../imtools/')
    sys.path.insert(0, '/home/tomas/projects/mrf_segmentation/')
    from mrfsegmentation.markov_random_field import MarkovRandomField

circle_in = None
circle_out = None
seeds_m = None
x_prev = 0
y_prev = 0


# mouse callback function
def process_mouse(event, x, y, flags, param):
    global circle_in, circle_out, x_prev, y_prev
    if event == cv2.EVENT_LBUTTONDOWN:
        if circle_in['active']:
            circle_in['x'] = x
            circle_in['y'] = y
        else:
            if not circle_in['visible']:
                circle_in['x'] = x
                circle_in['y'] = y
            circle_in['visible'] = True
            circle_in['active'] = True
            circle_in['width'] = 3

            circle_out['active'] = False
            circle_out['width'] = 1
    if event == cv2.EVENT_RBUTTONDOWN:
        if circle_out['active']:
            circle_out['x'] = x
            circle_out['y'] = y
        else:
            if not circle_out['visible']:
                circle_out['x'] = x
                circle_out['y'] = y
            circle_out['visible'] = True
            circle_out['active'] = True
            circle_out['width'] = 3

            circle_in['active'] = False
            circle_in['width'] = 1

    if event == cv2.EVENT_MOUSEMOVE:
        if flags == 1:
            circle_in['x'] = x
            circle_in['y'] = y
        elif flags == 2:
            circle_out['x'] = x
            circle_out['y'] = y
        elif flags == 4:
            if y < y_prev:
                delta = 2
            elif y > y_prev:
                delta = -2
            else:
                delta = 0
            x_prev = x
            y_prev = y

            if circle_in['active']:
                circle_in['radius'] += delta
            else:
                circle_out['radius'] += delta


def membrane_init(img):
    img_mark = img.copy()
    mark = np.zeros(img.shape[:2], np.uint8)

    sketch = Sketcher('mark membrane', [img_mark, mark], lambda: ((0, 0, 255), 255))
    while True:
        ch = cv2.waitKey()
        if ch == 27:
            break
        if ch == ord(' '):
            res = cv2.inpaint(img_mark, mark, 3, cv2.INPAINT_TELEA)
            cv2.imshow('inpaint', res)
        if ch == ord('r'):
            img_mark[:] = img
            mark[:] = 0
            sketch.show()
    cv2.destroyAllWindows()

    seeds_m = mark == 255
    print np.nonzero(seeds_m)
    plt.figure()
    plt.subplot(121), plt.imshow(img_mark)
    plt.subplot(122), plt.imshow(seeds_m)
    plt.show()


def mask_im(img):
    global circle_in, circle_out

    roi = np.zeros(img.shape[:2])
    cv2.circle(roi, (circle_out['x'], circle_out['y']), circle_out['radius'], 1, -1)

    seeds = np.zeros(img.shape[:2])
    cv2.circle(seeds, (circle_in['x'], circle_in['y']), circle_in['radius'], 1, -1)

    mask = roi + seeds

    # plt.figure()
    # plt.subplot(131), plt.imshow(mask, 'gray', interpolation='nearest')
    # plt.subplot(132), plt.imshow(roi, 'gray', interpolation='nearest')
    # plt.subplot(133), plt.imshow(seeds, 'gray', interpolation='nearest')
    # plt.show()

    return roi, seeds


def analyze_data(im, roi, seeds, seeds_m):
    if im.ndim == 3:
        img = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        img = im.copy()
    img = skires.denoise_bilateral(img, multichannel=False)
    roi = roi == 1
    seeds_o = seeds == 1
    seeds_b = skimor.binary_dilation(roi, skimor.disk(15)) - roi

    # img_c = img * roi
    # hist, bins = skiexp.histogram(img[np.nonzero(roi)])
    # hist_o, bins_o = skiexp.histogram(img[np.nonzero(seeds_o)])
    # hist_b, bins_b = skiexp.histogram(img[np.nonzero(seeds_b)])

    # segmentation_rw(img, seeds_o, seeds_b, seeds_m, roi)
    # segmentation_gc(img, seeds_o, seeds_b, seeds_m, roi)
    # segmentation_fcm(img, roi, n_clusters=3, show=True)

    misc(img, roi)
    # tt = TrackWin(img, skifil.frangi, 1, 10, 'frangi')

    # plt.figure()
    # plt.subplot(231), plt.imshow(roi, 'gray')
    # plt.subplot(232), plt.imshow(seeds_o, 'gray')
    # plt.subplot(233), plt.imshow(seeds_b, 'gray')
    # plt.subplot(234), plt.imshow(img_c, 'gray')
    #
    # plt.figure()
    # plt.plot(bins, hist, 'b-')
    # plt.plot(bins_o, hist_o, 'g-')
    # plt.plot(bins_b, hist_b, 'r-')
    # plt.show()


def segmentation_rw(img, seeds_o, seeds_b, seeds_m, roi):
    labels = -1 + roi + 2 * seeds_b + 2 * seeds_o + 3 * seeds_m
    probs = skiseg.random_walker(img, labels, beta=2000, return_full_prob=True)
    seg = np.argmax(probs, axis=0).astype(np.uint8)

    (cnts, _) = cv2.findContours(seg.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    im_vis = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    cv2.drawContours(im_vis, cnts, -1, (255, 0, 0), 2)

    plt.figure()
    plt.subplot(231), plt.imshow(img, 'gray')
    plt.subplot(232), plt.imshow(labels)
    plt.subplot(233), plt.imshow(seg)
    plt.subplot(234), plt.imshow(probs[0,...], 'gray')
    plt.subplot(235), plt.imshow(probs[1,...], 'gray')
    plt.subplot(236), plt.imshow(probs[2,...], 'gray')
    # plt.subplot(236), plt.imshow(im_vis)
    plt.show()


def segmentation_gc(img, seeds_o, seeds_b, seeds_m, roi):
    seeds = seeds_o + 2 * seeds_b + 3 * seeds_m
    scale = 1  # scaling parameter for resizing the image
    alpha = 1  # parameter for weighting the smoothness term (pairwise potentials)
    beta = 1  # parameter for weighting the data term (unary potentials)
    mrf = MarkovRandomField(img, seeds, mask=roi + seeds_b, alpha=alpha, beta=beta, scale=scale)
    # unaries = mrf.get_unaries()
    # mrf.set_unaries(unaries)

    labels = mrf.run()

    plt.figure()
    plt.subplot(221), plt.imshow(img, 'gray', interpolation='nearest'), plt.title('input image')
    plt.subplot(222), plt.imshow(seeds, interpolation='nearest'), plt.title('seeds')
    plt.subplot(223), plt.imshow(labels, interpolation='nearest'), plt.title('segmentation')
    plt.subplot(224), plt.imshow(skiseg.mark_boundaries(img, labels[2, :, :]), interpolation='nearest'), plt.title(
        'segmentation')
    plt.show()


def segmentation_fcm(img, mask=None, n_clusters=2, show=False, show_now=True, verbose=True):
    # set_verbose(verbose)

    if img.ndim == 2:
        is_2D = True
    else:
        is_2D = False

    if mask is None:
        mask = np.ones_like(img)

    coords = np.nonzero(mask)
    data = img[coords]
    # alldata = np.vstack((coords[0], coords[1], data))
    alldata = data.reshape((1, data.shape[0]))

    # _debug('Computing FCM...')
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(alldata, n_clusters, 2, error=0.005, maxiter=1000, init=None)

    # sorting the results in ascending order
    idx = np.argsort(cntr[:, 0])
    cntr = cntr[idx, :]
    u = u[idx, :]

    cm = np.argmax(u, axis=0) + 1  # cluster membership
    labels = np.zeros(img.shape)
    labels[coords] = cm

    # calculating cluster memberships
    mems = np.zeros(((n_clusters,) + img.shape))
    for i in range(n_clusters):
        cp = u[i, :] + 1  # cluster membership
        mem = np.zeros(img.shape)
        mem[coords] = cp
        mems[i,...] = mem

    if show:
        if is_2D:
            plt.figure()
            plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest'), plt.axis('off'), plt.title('input')
            plt.subplot(122), plt.imshow(labels, 'jet', interpolation='nearest'), plt.axis('off'), plt.title('fcm')
        # else:
            # py3DSeedEditor.py3DSeedEditor(labels).show()

            # u0 = np.zeros(img.shape)
            # u0[coords] = u[0]
            # u1 = np.zeros(img.shape)
            # u1[coords] = u[1]
            # d0 = np.zeros(img.shape)
            # d0[coords] = d[0]
            # d1 = np.zeros(img.shape)
            # d1[coords] = d[1]
            # plt.figure()
            # plt.subplot(231), plt.imshow(img, 'gray', interpolation='nearest'), plt.title('input')
            # plt.subplot(232), plt.imshow(u0, 'gray', interpolation='nearest'), plt.title('u0 - partititon matrix'), plt.colorbar()
            # plt.subplot(233), plt.imshow(u1, 'gray', interpolation='nearest'), plt.title('u1 - partititon matrix'), plt.colorbar()
            # plt.subplot(235), plt.imshow(d0, 'gray', interpolation='nearest'), plt.title('d0 - dist matrix'), plt.colorbar()
            # plt.subplot(236), plt.imshow(d1, 'gray', interpolation='nearest'), plt.title('d1 - dist matrix'), plt.colorbar()

        if show_now:
            plt.show()

    return mems, u, d, cntr, fpc


def misc(img, roi):
    # edgs = skifil.sobel(img, mask=roi)
    # real, imag = skifil.gabor(img, frequency=0.4)
    # real2, imag2 = skifil.gabor(img, frequency=0.8)
    frangi = skifil.frangi(img)
    hes = skifil.hessian(img)
    med = skifil.median(img, selem=skimor.disk(7))

    plt.figure()
    # plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(med, 'gray', interpolation='nearest')
    plt.subplot(131), plt.imshow(img, 'gray', interpolation='nearest')
    plt.subplot(132), plt.imshow(frangi, 'gray', interpolation='nearest')
    plt.subplot(133), plt.imshow(hes, 'gray', interpolation='nearest')

    # plt.subplot(221), plt.imshow(real, 'gray', interpolation='nearest')
    # plt.subplot(222), plt.imshow(imag, 'gray', interpolation='nearest')
    # plt.subplot(223), plt.imshow(real2, 'gray', interpolation='nearest')
    # plt.subplot(224), plt.imshow(imag2, 'gray', interpolation='nearest')

    # plt.figure()
    # plt.subplot(121), plt.imshow(img, 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(edgs, 'gray', interpolation='nearest')

    plt.show()


def init_test_cv(img):
    global circle_in, circle_out, seeds_m
    # im_vis = img
    # cv2.namedWindow('image', cv2.CV_GUI_NORMAL)#, cv2.WINDOW_NORMAL | cv2.WINDOW_AUTOSIZE)

    # initialize ROI and inner seeds
    if circle_in is None or circle_out is None:
        cv2.namedWindow('image', cv2.WINDOW_OPENGL + cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback('image', process_mouse)

        circle_in = {'x': 0, 'y': 0, 'radius': 50, 'color': (0, 255, 0), 'width': 1, 'visible': False, 'active': False}
        circle_out = {'x':0, 'y':0, 'radius':80, 'color':(0, 0, 255), 'width':1, 'visible':False, 'active':False}
        while True:
            im_vis = img.copy()
            if circle_in['visible']:
                cv2.circle(im_vis, (circle_in['x'], circle_in['y']), circle_in['radius'], circle_in['color'],
                           circle_in['width'])
            if circle_out['visible']:
                cv2.circle(im_vis, (circle_out['x'], circle_out['y']), circle_out['radius'], circle_out['color'],
                           circle_out['width'])
            cv2.imshow('image', im_vis)
            k = cv2.waitKey(50)
            print k

            if k & 0xFF == 13:  # enter
                print 'circle_in:', circle_in
                print 'circle_out:', circle_out
                break

            if k & 0xFF == 27:  # esc
                break
        cv2.destroyAllWindows()

    # initialize membrane seeds
    if seeds_m is None:
        membrane_init(img)

    # masking of the image
    roi, seeds = mask_im(img)

    # processing data
    analyze_data(img, roi, seeds, seeds_m)


if __name__ == '__main__':
    global circle_in, circle_out, seeds_m
    fname = '/home/tomas/Dropbox/Data/MemSkel/bug1_no_spline_cell.jpeg'

    # im = cv2.imread(fname)
    im = skiio.imread(fname)
    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    # img = skicol.rgb2gray(im)

    circle_in = {'color': (0, 255, 0), 'radius': 36, 'band': False, 'width': 1, 'y': 184, 'x': 197}
    circle_out = {'color': (0, 0, 255), 'radius': 120, 'band': False, 'width': 3, 'y': 211, 'x': 205}
    pts_mem = (np.array([116, 116, 116, 116, 117, 117, 117, 117, 117, 117, 117, 118, 118, 118, 118, 118, 118, 118, 119,
                         141, 141, 141, 142, 142, 142, 142, 142, 143, 143, 143, 143, 144, 144, 153, 153, 154, 154, 154,
                         154, 155, 155, 155, 155, 156, 156, 178, 178, 179, 179, 179, 179, 180, 180, 180, 180, 181, 181,
                         181, 182, 182, 182, 183, 184, 185, 185, 185, 186, 186, 186, 187, 187, 187, 188, 188, 188, 189,
                         189, 189, 189, 190, 190, 190, 190, 191, 191, 228, 228, 228, 228, 229, 229, 229, 229, 229, 229,
                         230, 230, 230, 230, 230, 230, 230, 231, 231, 231, 231, 231, 231, 231, 232, 232, 232, 246, 246,
                         246, 246, 246, 247, 247, 247, 247, 247, 247, 247, 247, 247, 247, 248, 248, 248, 248, 248, 248,
                         248, 248, 248, 248, 248, 248, 248, 248, 248, 249, 249, 249, 249, 249, 249, 249, 249, 249, 249,
                         249, 249, 249, 249, 249, 250, 250, 250, 250, 250, 250, 250, 250, 250, 251, 251, 251, 251]),
               np.array([182, 183, 184, 185, 180, 181, 182, 183, 184, 185, 186, 179, 180, 181, 182, 183, 184, 185, 180,
                         250, 251, 252, 249, 250, 251, 252, 253, 250, 251, 252, 253, 251, 252, 126, 127, 125, 126, 127,
                         128, 125, 126, 127, 128, 126, 127, 117, 118, 116, 117, 118, 119, 116, 117, 118, 119, 117, 118,
                         119, 117, 118, 119, 118, 279, 278, 279, 280, 278, 279, 280, 278, 279, 280, 278, 279, 280, 278,
                         279, 280, 281, 278, 279, 280, 281, 279, 280, 138, 139, 140, 141, 137, 138, 139, 140, 141, 142,
                         137, 138, 139, 140, 141, 142, 143, 138, 139, 140, 141, 142, 143, 144, 141, 142, 143, 210, 211,
                         212, 279, 280, 209, 210, 211, 212, 213, 214, 278, 279, 280, 281, 210, 211, 212, 213, 214, 215,
                         216, 217, 218, 219, 277, 278, 279, 280, 281, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220,
                         276, 277, 278, 279, 280, 216, 217, 218, 219, 220, 276, 277, 278, 279, 218, 219, 277, 278]))
    seeds_m = np.zeros(im.shape[:2], dtype=np.uint8)
    seeds_m[pts_mem] = 1

    # plt.figure()
    # plt.imshow(seeds_m, 'gray', interpolation='nearest')
    # plt.show()

    init_test_cv(im)

    # plt.figure()
    # plt.imshow(img, 'gray', interpolation='nearest')
    # plt.show()