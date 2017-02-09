from __future__ import division

import cv2
import skimage.filters as skifil
import skimage.io as skiio
import skimage.restoration as skires
import skimage.exposure as skiexp

import matplotlib.pyplot as plt


def update(value):

    # range1 = cv2.getTrackbarPos('range1', winname)
    # range2 = cv2.getTrackbarPos('range2', winname)
    # scale_step = cv2.getTrackbarPos('scale_step')
    # beta1 = cv2.getTrackbarPos('beta1') / 10
    # beta2 = cv2.getTrackbarPos('beta2') / 10
    pass


def run(img, winname='frangi'):

    cv2.namedWindow(winname)
    cv2.createTrackbar('range1', winname, 1, 100, update)
    cv2.createTrackbar('range2', winname, 1, 100, update)
    cv2.createTrackbar('scale_step', winname, 1, 10, update)
    cv2.createTrackbar('beta1', winname, 0, 200, update)
    cv2.createTrackbar('beta2', winname, 0, 200, update)

    cv2.setTrackbarPos('range1', winname, 1)
    cv2.setTrackbarPos('range2', winname, 10)
    cv2.setTrackbarPos('scale_step', winname, 2)
    cv2.setTrackbarPos('beta1', winname, 5)
    cv2.setTrackbarPos('beta2', winname, 150)

    im_vis = img.copy()

    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(img, 'gray')
    # plt.subplot(122)
    # plt.imshow(skifil.frangi(img), 'gray')
    # plt.show()
    cv2.imshow('input', img)

    while True:
        k = cv2.waitKey(50)
        if k & 0xFF == 27:  # esc
            break

        range1 = cv2.getTrackbarPos('range1', winname)  # 0
        range2 = cv2.getTrackbarPos('range2', winname)  # 4
        scale_step = cv2.getTrackbarPos('scale_step', winname)  # 2
        beta1 = cv2.getTrackbarPos('beta1', winname) / 10  # 0.5
        beta2 = cv2.getTrackbarPos('beta2', winname) / 10  # 15

        im_vis = skifil.frangi(img, (range1, range2), scale_step, beta1, beta2)
        cv2.imshow(winname, skiexp.rescale_intensity(im_vis, out_range=(0, 1)))

    cv2.destroyAllWindows()

if __name__ == '__main__':
    fname = '/home/tomas/Dropbox/Data/MemSkel/bug1_no_spline_cell.jpeg'
    img = cv2.imread(fname, 0)
    img = skires.denoise_bilateral(img, multichannel=False)

    run(img)