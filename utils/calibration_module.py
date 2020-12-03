"""
This is the script containing the calibration module, basically calculating homography matrix.

This code and data is released under the Creative Commons Attribution-NonCommercial 4.0 International license (CC BY-NC.) In a nutshell:
    # The license is only for non-commercial use (commercial licenses can be obtained from Stanford).
    # The material is provided as-is, with no warranties whatsoever.
    # If you publish any code, data, or scientific work based on this, please cite our work.

Technical Paper:
Y. Peng, S. Choi, N. Padmanaban, G. Wetzstein. Neural Holography with Camera-in-the-loop Training. ACM TOG (SIGGRAPH Asia), 2020.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


def circle_detect(captured_img, num_circles, spacing, pad_pixels=(0., 0.), show_preview=True):
    """
    Detects the circle of a circle board pattern

    :param captured_img: captured image
    :param num_circles: a tuple of integers, (num_circle_x, num_circle_y)
    :param spacing: a tuple of integers, in pixels, (space between circles in x, space btw circs in y direction)
    :param show_preview: boolean, default True
    :param pad_pixels: coordinate of the left top corner of warped image.
                       Assuming pad this amount of pixels on the other side.
    :return: a tuple, (found_dots, H)
             found_dots: boolean, indicating success of calibration
             H: a 3x3 homography matrix (numpy)
    """

    # Binarization
    # org_copy = org.copy() # Otherwise, we write on the original image!
    img = (captured_img.copy() * 255).astype(np.uint8)
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.medianBlur(img, 15)
    img_gray = img.copy()

    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 121, 0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = 255 - img

    # Blob detection
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.filterByColor = True
    params.minThreshold = 128

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 50

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.785

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = False
    params.minInertiaRatio = 0.01

    detector = cv2.SimpleBlobDetector_create(params)

    # Detecting keypoints
    # this is redundant for what comes next, but gives us access to the detected dots for debug
    keypoints = detector.detect(img)
    found_dots, centers = cv2.findCirclesGrid(img, num_circles,
                                              blobDetector=detector, flags=cv2.CALIB_CB_SYMMETRIC_GRID)

    # Drawing the keypoints
    cv2.drawChessboardCorners(captured_img, num_circles, centers, found_dots)
    img_gray = cv2.drawKeypoints(img_gray, keypoints, np.array([]), (0, 255, 0),
                                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Find transformation
    H = np.array([[1., 0., 0.],
                  [0., 1., 0.],
                  [0., 0., 1.]], dtype=np.float32)
    if found_dots:
        # Generate reference points to compute the homography
        ref_pts = np.zeros((num_circles[0] * num_circles[1], 1, 2), np.float32)
        pos = 0
        for i in range(0, num_circles[1]):
            for j in range(0, num_circles[0]):
                ref_pts[pos, 0, :] = spacing * np.array([j, i]) + np.array(pad_pixels)
                pos += 1

        H, mask = cv2.findHomography(centers, ref_pts, cv2.RANSAC, 1)
        if show_preview:
            dsize = [int((num_circs - 1) * space + 2 * pad_pixs)
                     for num_circs, space, pad_pixs in zip(num_circles, spacing, pad_pixels)]
            captured_img_warp = cv2.warpPerspective(captured_img, H, tuple(dsize))

    if show_preview:
        fig = plt.figure()

        ax = fig.add_subplot(223)
        ax.imshow(img_gray, cmap='gray')

        ax2 = fig.add_subplot(221)
        ax2.imshow(img, cmap='gray')

        ax3 = fig.add_subplot(222)
        ax3.imshow(captured_img, cmap='gray')

        if found_dots:
            ax4 = fig.add_subplot(224)
            ax4.imshow(captured_img_warp, cmap='gray')

        plt.show()

    return found_dots, H


class Calibration:
    def __init__(self, num_circles=(21, 12), spacing_size=(80, 80), pad_pixels=(0, 0)):
        self.num_circles = num_circles
        self.spacing_size = spacing_size
        self.pad_pixels = pad_pixels
        self.h_transform = np.array([[1., 0., 0.],
                                     [0., 1., 0.],
                                     [0., 0., 1.]])

    def calibrate(self, img, show_preview=True):
        found_corners, self.h_transform = circle_detect(img, self.num_circles,
                                                        self.spacing_size, self.pad_pixels, show_preview)
        return found_corners

    def get_transform(self):
        return self.h_transform

    def __call__(self, input_img, img_size=None):
        """
        This forward pass returns the warped image.

        :param input_img: A numpy grayscale image shape of [H, W].
        :param img_size: output size, default None.
        :return: output_img: warped image with pre-calculated homography and destination size.
        """

        if img_size is None:
            img_size = [int((num_circs - 1) * space + 2 * pad_pixs)
                        for num_circs, space, pad_pixs in zip(self.num_circles, self.spacing_size, self.pad_pixels)]
        output_img = cv2.warpPerspective(input_img, self.h_transform, tuple(img_size))

        return output_img
