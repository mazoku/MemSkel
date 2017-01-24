import cv2


class TrackWin:

    def __init__(self, im, fcn, min_param_val, max_param_val, winname='window'):
        self.im = im
        self.im_vis = im.copy()
        self.fcn = fcn
        self.min_param_val = min_param_val
        self.max_param_val = max_param_val
        self.winname = winname

        cv2.namedWindow(self.winname)
        cv2.createTrackbar('tbar', self.winname, self.min_param_val, self.max_param_val, self.update)

        while True:
            k = cv2.waitKey(50)
            cv2.imshow(self.winname, self.im_vis)
            if k & 0xFF == 27:  # esc
                break
        cv2.destroyAllWindows()

    def update(self, value):
        self.im_vis = self.fcn(self.im, value)
        cv2.imshow(self.winname, self.im_vis)

        # 0xFF & cv2.waitKey()
        # cv2.destroyAllWindows()