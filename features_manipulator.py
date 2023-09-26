import cv2 as cv
import numpy as np

from image_data import  ImageData

class FeaturesManipulator:
    def __init__(self):
        self.sift = cv.SIFT_create()

        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)

        # FLANN based matcher with implementation of k nearest neighbour.
        self._matcher = cv.FlannBasedMatcher(index_params, search_params)

    def get_matches(self, kp1, des1, kp2, des2):
        matches_rough = filter(lambda m: m[0].distance < m[1].distance * 0.7, self._matcher.knnMatch(des1, des2, k = 2))
        matches = map(lambda m:m[0], matches_rough)
        matches = sorted(matches, key = lambda m: m.distance)

        image_idx1 = np.array([m.queryIdx for m in matches])
        image_idx2 = np.array([m.trainIdx for m in matches])

        kp1_aligned = (np.array(kp1))[image_idx1]
        kp2_aligned = (np.array(kp2))[image_idx2]

        image_points1 = np.array([kp.pt for kp in kp1_aligned])
        image_points2 = np.array([kp.pt for kp in kp2_aligned])

        self._matcher.clear()

        return image_points1, image_points2, image_idx1, image_idx2

    def get_features(self, image_mat):
        image_mat = image_mat[:,:,::-1]
        return self.sift.detectAndCompute(image_mat, None)

    def find_3d_matches(self, kp, des, cahe: dict[ImageData]):
        kps, descs = [], []

        prev_images_count = 0
        for image_name in cahe.keys():
            image_data = cahe[image_name]
            kps.append(image_data.kp)
            descs.append(image_data.des)
            prev_images_count += 1

        self._matcher.add(descs)
        self._matcher.train()

        matches_3d_2d = self._matcher.match(queryDescriptors = des)

        self._matcher.clear()

        matches_3d_2d = sorted(matches_3d_2d, key = lambda m: m.distance)

        posible_good_lenght = int(len(matches_3d_2d) * (3.0 / prev_images_count))
        matches_threshold = 12
        lenght = posible_good_lenght if posible_good_lenght > matches_threshold else matches_threshold
        matches_3d_2d = matches_3d_2d[: lenght]

        return matches_3d_2d