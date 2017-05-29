from __future__ import division
__author__ = 'Ryba'

# TODO's ------------------------------
#   TODO: zkontrolovat funkcnost pomoci __main__
# TODO's ------------------------------

import numpy as np
import skimage.morphology as skimor
import skimage.exposure as skiexp
from constants import *
import cv2
import cPickle as pickle


class Segmentator(object):

    def __init__(self):
        self.data = None
        self.ball_radius = 3  # radius defining the size of neighborhood used for calculations
        self.define_ball_mask()
        self.min_diff = 10  # if the difference between distances to inner and outer masks are smaller than this, accept the newbie
        self.localization_w = .2  # weight of localization, <0, 1>; the higher the value the more localized the calculation
        self.draw_steps = 5  # number of iterations after which the figures are updated
        self.max_iterations = 1000  # maximal number of iterations
        self.strel = np.ones((3, 3), dtype=np.int)  # structure element of dilatation for getting neighboring points
        self.model_hist = None  # histogram model for back projection

    @property
    def ball_radius(self):
        return self.__ball_radius

    @ball_radius.setter
    def ball_radius(self, x):
        self.__ball_radius = x
        self.define_ball_mask()

    def get_threshold(self, img, pt=70, t=None):
        '''
        Thresholding - default method is percentile.
        @param pt: percentile threshold, default is 70%
        @type pt: int
        @param t: threshold value, default is None. If it is not None, then this value is used prior to the
        percentile thresholding.
        @type t: float
        @return: thresholded image
        @rtype: ndarray, dtype=bool
        '''

        if img is not None:
            if t is None:
                # percent thresh
                hist, bins = skiexp.histogram(img, nbins=100)
                hist = hist / hist.sum()
                # hist = hist.astype(np.float) / hist.sum()
                cum_hist = hist.copy()
                for i in range(1, len(cum_hist)):
                    cum_hist[i] += cum_hist[i - 1]

                # diff = cum_hist - (pt / 100)
                # diff *= diff > 0
                #
                # t_ind = np.nonzero(diff)[0][0]
                t_ind = np.where(cum_hist > pt / 100.)[0][0]
                t = bins[t_ind]
            # mask = img > t
            return t

    def define_ball_mask(self):
        self.ball_mask = np.zeros((2 * self.ball_radius + 1, 2 * self.ball_radius + 1, 2), dtype=np.int)
        self.ball_mask[:, :, 1] = np.tile(np.arange(- self.ball_radius, self.ball_radius + 1), [2 * self.ball_radius + 1, 1])
        self.ball_mask[:, :, 0] = self.ball_mask[:, :, 1].conj().transpose()

    def segment(self, slice_idx, update_fcn=None, update_rate=5, progress_fig=False):
        '''
        Run the segmentation process until convergence or max. number of iterations is achieved.
        :param slice_idx: index of current slice
        :param update_rate: number of iterations until figures update
        :param progress_fig: whether to show independent visualization window
        :return: segmentation ndarray
        '''
        it = 0
        changed = True
        segmentation = self.data.seeds[slice_idx, ...] * self.data.roi[slice_idx, ...]
        newbies = segmentation.copy()
        while changed and it < self.max_iterations:
            it += 1
            segmentation_new, accepted, refused, newbies = self.segmentation_step(self.data.image[slice_idx, ...], newbies,
                                                                                  segmentation.copy(), self.data.roi[slice_idx, ...])
            newbies = segmentation_new - segmentation
            segmentation = segmentation_new
            if progress_fig:
                cnts = cv2.findContours(segmentation.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]
                cnt = sorted(cnts, key=cv2.contourArea)[-1]
                img_vis = cv2.cvtColor(self.data.image[slice_idx, ...], cv2.COLOR_GRAY2BGR)
                cv2.drawContours(img_vis, [cnt], -1, (0, 0, 255), 1)
                cv2.imshow('segmentation', img_vis)
                cv2.waitKey(50)

            if not accepted.any():
                changed = False
                segmentation = skimor.binary_closing(segmentation, selem=np.ones((5, 5), dtype=np.int))

            if it % update_rate == 0:
                update_fcn(segmentation)

        if progress_fig:
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        self.data.segmentation[slice_idx, ...] = segmentation
        return segmentation

    def segmentation_step(self, img, newbies, segmentation, roi):
        """
        Performs one iteration of segmentation.
        :param img: input image
        :param segmentation: current segmentation mask
        :param roi: region of interest defined by thresholding, for example
        :return: updated segmentation mask, list of accepted and refused points
        """
        # get new candidates by dilating mask of newbies
        dilated_mask = skimor.binary_dilation(newbies, self.strel)
        candidates = (dilated_mask - newbies) * roi
        candidates = np.argwhere(candidates)

        # lists of filtered candidates
        accepted = list()
        refused = list()

        # mean intensity value of current segmentation
        obj_ints = img[np.nonzero(segmentation)]
        mean_obj_int = obj_ints.mean()

        # iterate over candidates
        for c in candidates:
            # for each pt in ball mask placed on the newbie, identify pts lying inside segmentation mask (inners) and
            # outside segmentation mask (outers)
            inners, outers = self.mask_newbie(c, segmentation)

            # get distance of the newbie to the mean intensity of inners and outers and to the mean intensity of segmentation
            dist_in, dist_out = self.get_distances(img, c, inners, outers)
            dist_mean_obj = np.absolute(mean_obj_int - img[c[0], c[1]])

            # localization weighting - the higher the alpha the more localized the calculation
            dist_in_w = self.localization_w * dist_in + (1 - self.localization_w) * dist_mean_obj

            # accept the candidate if its distance is closer to the dist_in_w or
            # the difference between dist_in and dist_out is smaller than self.min_diff
            if dist_in_w < dist_out or np.absolute(dist_in - dist_out) < self.min_diff:
                segmentation[c[0], c[1]] = True
                accepted.append(c)
            else:
                refused.append(c)

        accepted = np.array(accepted)
        refused = np.array(refused)

        return segmentation, accepted, refused, newbies

    def calc_model(self, img, mask, show=False, show_now=True):
        self.model_hist = cv2.calcHist([img], [0,], mask, [64], [0, 256])
        cv2.normalize(self.model_hist, self.model_hist, 0, 255, cv2.NORM_MINMAX)

        if show:
            h = np.zeros((300, 256, 3))
            hist = np.int32(np.around(self.model_hist))
            for x, y in enumerate(hist):
                cv2.line(h, (x, 0), (x, y), (255, 255, 255))
            y = np.flipud(h)
            cv2.imshow('hist model', y)
            if show_now:
                cv2.waitKey(0)

    def segment_stack(self, idx):
        init_membrane = self.data.segmentation[idx, ...]
        init_seeds = self.data.seeds[idx, ...]

        # create histogram model for backprojection
        # self.calc_model(self.data.image[idx, ...],  init_membrane)
        self.calc_model(self.data.image[idx, ...],  init_seeds, show=False)

        # find seeds in other slices
        for i in range(self.data.n_slices):
            # skip the initial frame
            if i == idx:
                continue

            # grab the slice
            im = self.data.image[i, ...] * init_membrane

            # pouze seedy
            print 'Propagating seeds ...',
            bp = cv2.calcBackProject([im], [0,], self.model_hist, [0, 256], 1)

            cv2.imshow('backproj', np.hstack((im, bp)))
            cv2.waitKey(0)
            print 'done'

            # membrana
            print ''



        # segment each slice

    def mask_newbie(self, newbie, mask):
        """
        Places the ball mask on the newbie and identify all points of the ball that belong
        to the mask (inners) and that belong not to the mask (outers)
        :param newbie: point where the mask would be placed
        :param mask: current segmentation
        :return: inners = list of pts (ball mask and mask), outers = list of pts (ball mask and not mask)
        """
        ball_coords = self.ball_mask.copy()
        ball_coords[:, :, 0] += newbie[0]
        ball_coords[:, :, 1] += newbie[1]

        #    warnings.warn( 'Place for possible speeding up the process.' )
        #    if np.amin( ballM[:,:,0] ) < 0:
        try:
            while np.amin(ball_coords[:, :, 0]) < 0:
                ball_coords = ball_coords[1:, :, :]
            while np.amin(ball_coords[:, :, 1]) < 0:
                ball_coords = ball_coords[:, 1:, :]
            while np.amax(ball_coords[:, :, 0]) >= mask.shape[0]:
                ball_coords = ball_coords[:-1, :, :]
            while np.amax(ball_coords[:, :, 1]) >= mask.shape[1]:
                ball_coords = ball_coords[:, :-1, :]
        except ValueError:
            self.statusbar.config(text='Error while controlling ball boundaries.')

        masked = mask[ball_coords[:, :, 0], ball_coords[:, :, 1]]
        inners = np.nonzero(masked)
        masked[self.ball_radius, self.ball_radius] = True  # exclude center from computation
        outers = np.nonzero(np.invert(masked))

        inners = ball_coords[inners[0], inners[1], :]
        outers = ball_coords[outers[0], outers[1], :]

        return inners, outers

    def get_distances(self, im, newbie, inners, outers):
        """
        Calculates intensity difference between the newbie and mean value of inners and outers, respectively
        :param im: input image
        :param newbie: point
        :param inners: list of points belonging to the ball mask and the ROI
        :param outers: list of points belonging to the ball mask and not to the ROI
        :return: two distances
        """
        int_in = im[inners[:, 0], inners[:, 1]]
        int_out = im[outers[:,0], outers[:, 1]]
        int_new = im[newbie[0], newbie[1]]

        mean_in = int_in.mean()
        mean_out = int_out.mean()

        dist_in = np.absolute(mean_in - int_new)
        dist_out = np.absolute(mean_out - int_new)

        return dist_in, dist_out


def example_immunohisto():
    img = skidat.immunohistochemistry()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, None, fx=0.5, fy=0.5)

    # get seeds -----------
    c = (155, 191)
    # c = (110, 191)
    rad = 10
    if c == (0, 0):
        done = False

        def mouse_callback(event, x, y, flags, param):
            global c, done
            if event == cv2.EVENT_MOUSEMOVE:
                c = tuple((x, y))
            if event == cv2.EVENT_LBUTTONDOWN:
                c = tuple((x, y))
                print 'clicked: {}'.format(c)
                done = True
            update()

        def update():
            global c, img
            img_vis = cv2.circle(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), c, rad, (0, 0, 255), 2)
            cv2.imshow('img', img_vis)

        cv2.namedWindow('img')
        cv2.imshow('img', img)
        cv2.setMouseCallback('img', mouse_callback)
        while not done:
            update()
            cv2.waitKey(10)

    seeds = np.zeros(img.shape, dtype=np.uint8)
    cv2.circle(seeds, c, rad, 1, -1)

    # cv2.imshow('seeds', np.hstack((img, seeds)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # ----------------

    img = np.expand_dims(img, 0)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    data = ImageData()
    data.image = img
    data.segmentation = np.zeros(img.shape, dtype=np.uint8)
    # data.seeds = np.zeros(img.shape, dtype=np.uint8)
    data.seeds = np.expand_dims(seeds, 0)
    data.roi = np.ones(img.shape, dtype=np.uint8)
    data.n_slices, data.n_rows, data.n_cols = img.shape

    segmentator = Segmentator()
    segmentator.data = data
    segmentator.segment(0, progress_fig=True)


def example_membrane():
    data_pkl = pickle.load(open('data/test_data_3d.pickle', 'rb'))
    data = ImageData()
    data.image = data_pkl['im']
    data.segmentation = data_pkl['segmentation']
    data.seeds = data_pkl['seeds']
    data.roi = data_pkl['roi']
    idx = data_pkl['idx']
    data.n_slices = data.segmentation.shape[0]

    segmentator = Segmentator()
    segmentator.data = data

    cv2.imshow('image', np.hstack(tuple([x for x in data.image])))
    cv2.imshow('membrane', np.hstack(tuple([255 * x for x in data.segmentation])))
    cv2.imshow('seeds', np.hstack(tuple([255 * x for x in data.seeds])))
    cv2.waitKey(0)

    segmentator.segment_stack(idx)


if __name__ == '__main__':
    from imagedata import ImageData
    import skimage.data as skidat
    import cv2

    example_membrane()