import cv2 as cv
import numpy as np

from image_data import  ImageData
from features_manipulator import  FeaturesManipulator
from pair_data import  PairData

class PoseEstimator:
    cache:dict[ImageData]
    features_manipulator:FeaturesManipulator
    K = None

    def __init__(self, cache:dict[ImageData], features_manipulator:FeaturesManipulator, K):
        self.cache = cache
        self.features_manipulator = features_manipulator
        self.K = K

    def initial_run(self, image_name1:str, image_name2:str):
        image1_mat = cv.imread(image_name1)
        image2_mat = cv.imread(image_name2)
        kp1, des1 = self.features_manipulator.get_features(image1_mat)
        kp2, des2 = self.features_manipulator.get_features(image2_mat)

        image_data1 = self.cache[image_name1] = ImageData(kp1, des1)
        image_data2 = self.cache[image_name2] = ImageData(kp2, des2)

        image_points1, image_points2, image_idx1, image_idx2 = self.features_manipulator.get_matches(kp1, des1, kp2, des2)

        F, mask = cv.findFundamentalMat(image_points1, image_points2, cv.FM_RANSAC, 0.1, 0.99)

        mask = mask.astype(bool).flatten()

        E = self.K.T.dot(F.dot(self.K))
        _, R, t, _ = cv.recoverPose(E, image_points1[mask], image_points2[mask], self.K)

        image_data1.R = np.eye(3, 3)
        image_data1.t = np.zeros((3,1))
        image_data1.refs = np.ones((len(kp1),)) * -1

        image_data2.R = R
        image_data2.t = t
        image_data2.refs = np.ones((len(kp2),)) * -1

        return  R, t, PairData(
            image_points1 = image_points1[mask],
            image_points2 = image_points2[mask],
            image_idx1 = image_idx1[mask],
            image_idx2 = image_idx2[mask]
        )

    def run(self, image_name, point_cloud, image_names):
        image_mat = cv.imread(image_name)
        kp, des = self.features_manipulator.get_features(image_mat)

        matches_3d_2d = self.features_manipulator.find_3d_matches(kp, des, self.cache)

        points_3d, points_2d = np.zeros((0, 3)), np.zeros((0, 2))

        for m in matches_3d_2d:
            old_image_idx, old_image_des_idx, image_des_idx = m.imgIdx, m.trainIdx, m.queryIdx

            old_image_name = image_names[old_image_idx]
            point_cloud_idx = self.cache[old_image_name].refs[old_image_des_idx]

            if point_cloud_idx >= 0:
                point_3d = point_cloud[int(point_cloud_idx)]
                points_3d = np.concatenate((points_3d, point_3d[np.newaxis]), axis = 0)

                point_2d = np.array(kp[int(image_des_idx)].pt)
                points_2d = np.concatenate((points_2d, point_2d[np.newaxis]), axis = 0)

        _, R, t, _ = cv.solvePnPRansac(points_3d[:,np.newaxis], points_2d[:,np.newaxis], self.K, None, confidence=.99,flags=cv.SOLVEPNP_DLS, reprojectionError = 8.)

        R,_ = cv.Rodrigues(R)
        image_data = self.cache[image_name] = ImageData(kp, des)
        image_data.R, image_data.t, image_data.refs = R, t, np.ones((len(kp),)) * -1

        return R, t, matches_3d_2d
