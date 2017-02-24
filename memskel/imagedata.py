import cv2
import numpy as np
import matplotlib.pyplot as plt
from libtiff import TIFFimage
from PIL import Image


class ImageData:

    def __init__(self):
        self.image = None  # image data to be segmented (usually TIFF format)
        self.segmentation = None  # membrane mask - outcome of segmentation
        self.seeds = None  # seed points defined by user (maybe also automatically?) used for segmentation
        self.roi = None  # region of interest; reduce the area where the segmentation is done
        self.spline = None  # approximation of the membrane mask; medial axis
        self.n_rows = None
        self.n_cols = None
        self.n_slices = None

    def load(self, fname):
        # self.data = self.pilImg2npArray(Image.open(path))
        self.image = self.pilImg2npArray(Image.open(fname)).astype(np.uint8)
        # self.data = cv2.imread(fname)
        # self.data = cv2.imreadmulti(fname)
        self.segmentation = np.zeros(self.image.shape, dtype=np.uint8)
        self.seeds = np.zeros(self.image.shape, dtype=np.uint8)
        self.roi = np.zeros(self.image.shape, dtype=np.uint8)
        self.n_slices, self.n_rows, self.n_cols = self.image.shape

    def pilImg2npArray(self, im):
        page = 0
        try:
            while 1:
                im.seek(page)
                page = page + 1
        except EOFError:
            pass

        numFrames = page
        im.seek(0)
        imArray = np.zeros((numFrames, np.array(im).shape[0], np.array(im).shape[1]))
        for i in range(numFrames):
            im.seek(i)
            imArray[i, :, :] = np.array(im.convert('L'))
        return imArray


class DataViewer:

    def __init__(self, data, scale=None):
        self.image = data.astype(np.uint8)
        self.curr_idx = 0
        self.title = 'viewer'
        self.scale = scale

    def show(self):
        cv2.namedWindow(self.title)
        # cv2.imshow(self.title, self.data[self.curr_idx, ...])
        self.disp_img()
        cv2.createTrackbar('page', self.title, 0, self.image.shape[0] - 1, self.update)
        # cv2.setMouseCallback(self.title, self.update)

        while 1:
            if cv2.waitKey(20) & 0xFF == 27:
                break
        cv2.destroyAllWindows()

    def disp_img(self):
        if self.scale is not None:
            im = cv2.resize(self.image[self.curr_idx, ...], None, fx=self.scale, fy=self.scale)
        else:
            im = self.image[self.curr_idx, ...]
        cv2.imshow(self.title, im)

    def update(self, value):
        self.curr_idx = value
        # cv2.imshow(self.title, self.data[self.curr_idx, ...])
        self.disp_img()

if __name__ == '__main__':
    fname = 'data/smallStack.tif'
    # fname = 'data/mujTifStack.tif'

    data = ImageData()
    data.load(fname)

    DataViewer(data.image, scale=4).show()
    # for d in data.data:
    #     plt.subplot(1, )

    pass