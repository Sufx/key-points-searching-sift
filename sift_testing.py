import numpy as np
import cv2
from cv2.xfeatures2d import SIFT_create

class SIFT:

    def __init__(self, file):
        """
        The constructor
        """
        self.img = cv2.imread(file)

    def add_noise(self, sigma):

        yuv = cv2.cvtColor(self.img, cv2.COLOR_BGR2YUV)
        yuv[..., 0] = yuv[..., 0] + np.random.normal(0, np.sqrt(sigma), yuv[...,0].shape)
        yuv[..., 0] = np.clip(yuv[..., 0], 0, 255)
        noise_img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return noise_img

    def getKpAndDescriptors(self, img, detector="sift"):


        if detector.lower() == "harris":
            thr = 0.01
            size = 2
            dst = cv2.cornerHarris(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 2, 3, 0.04)
            dst = cv2.dilate(dst, None)
            kp = np.argwhere(dst > thr * dst.max())
            key_points = [cv2.KeyPoint(k[0], k[1], 2) for k in kp]
            sift_creator = SIFT_create()
            __, descriptors = sift_creator.compute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), key_points)
            return descriptors[:100]
        elif detector.lower() == "sift":
            sift_creator = SIFT_create()
            kp, descriptors = sift_creator.detectAndCompute(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), None)
            return descriptors[:100]

