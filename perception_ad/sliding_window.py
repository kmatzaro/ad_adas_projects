import numpy as np
import matplotlib.pyplot as plt

class SlidingWindowSearcher:
    def __init__(self, config):
        self.config    = config['sliding_window']
        self.nwindows  = self.config['nwindows']
        self.margin    = self.config['margin']
        self.minpix    = self.config['minpix']

    def search(self, binary_warped: np.ndarray, return_debug=False):
        # 1) Histogram
        histogram    = np.sum(binary_warped[binary_warped.shape[0]//2:], axis=0)
        midpoint     = histogram.shape[0] // 2
        leftx_base   = np.argmax(histogram[:midpoint])
        rightx_base  = np.argmax(histogram[midpoint:]) + midpoint

        # 2) Prepare
        window_height   = binary_warped.shape[0] // self.nwindows
        nonzero         = binary_warped.nonzero()
        nonzeroy, nonzerox = nonzero
        leftx_current   = leftx_base
        rightx_current  = rightx_base
        left_lane_inds  = []
        right_lane_inds = []
        window_boxes    = []   # << record each window’s coords

        # 3) Slide
        for win in range(self.nwindows):
            y_low  = binary_warped.shape[0] - (win+1)*window_height
            y_high = binary_warped.shape[0] -  win   *window_height
            xll    = leftx_current  - self.margin
            xlh    = leftx_current  + self.margin
            xrl    = rightx_current - self.margin
            xrh    = rightx_current + self.margin

            # record boxes as ((x1,y1),(x2,y2)) pairs
            window_boxes.append((
              (xll, y_low,  xlh, y_high),
              (xrl, y_low,  xrh, y_high)
            ))

            # find pixels in box
            good_left_inds  = ((nonzeroy>=y_low)&(nonzeroy<y_high)&
                               (nonzerox>=xll)&(nonzerox< xlh)).nonzero()[0]
            good_right_inds = ((nonzeroy>=y_low)&(nonzeroy<y_high)&
                               (nonzerox>=xrl)&(nonzerox< xrh)).nonzero()[0]

            left_lane_inds .append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # recenter
            if len(good_left_inds)  > self.minpix:
                leftx_current  = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > self.minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # 4) flatten
        if left_lane_inds:
            left_lane_inds  = np.concatenate(left_lane_inds)
        else:
            left_lane_inds  = np.array([], dtype=int)
        if right_lane_inds:
            right_lane_inds = np.concatenate(right_lane_inds)
        else:
            right_lane_inds = np.array([], dtype=int)

        # 5) extract pixel coords
        leftx,  lefty  = nonzerox[left_lane_inds],  nonzeroy[left_lane_inds]
        rightx, righty = nonzerox[right_lane_inds], nonzeroy[right_lane_inds]

        if return_debug:
            return (leftx, lefty, rightx, righty,
                    nonzerox, nonzeroy, window_boxes)
        else:
            return leftx, lefty, rightx, righty
