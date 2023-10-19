import numpy as np
import cv2 as cv

from image_data import  ImageData
from pair_data import  PairData
from utility_fucntions import error_mini

class PointCloudGenerator:
    def __init__(self, K):
        self.K = K
        self.K_Inv = np.linalg.inv(self.K)
        self.point_cloud = np.zeros((0, 3))

    def optimised_point_cloud_mask(self):
        dists = []
        def dist(a, b):
            return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2

        total_sum_dist = 0
        for i in range(0, len(self.point_cloud)):
            sum_dist = 0
            for j in range(len(self.point_cloud)):
                if i != j:
                    sum_dist += dist(self.point_cloud[i], self.point_cloud[j])
            dists.append(sum_dist)
            total_sum_dist += sum_dist
        average_dist = float(total_sum_dist) / len(self.point_cloud)
        return  np.array([dist < average_dist * 3 for dist in dists]).astype(bool)

    def initial_render(self, image_data1: ImageData, image_data2: ImageData, pair_data: PairData):
        image_points_hom1 = cv.convertPointsToHomogeneous(pair_data.image_points1)[:, 0, :]
        image_points_hom2 = cv.convertPointsToHomogeneous(pair_data.image_points2)[:, 0, :]

        image_points_norm1 = (self.K_Inv.dot(image_points_hom1.T)).T
        image_points_norm2 = (self.K_Inv.dot(image_points_hom2.T)).T

        image_points_norm1 = cv.convertPointsFromHomogeneous(image_points_norm1)[:, 0, :]
        image_points_norm2 = cv.convertPointsFromHomogeneous(image_points_norm2)[:, 0, :]

        points_4d = cv.triangulatePoints(
            np.hstack((image_data1.R,image_data1.t)),
            np.hstack((image_data2.R, image_data2.t)),
            image_points_norm1.T,
            image_points_norm2.T
        )

        points_3d = cv.convertPointsFromHomogeneous(points_4d.T)[: , 0, :]
        self.point_cloud = np.concatenate((self.point_cloud, points_3d), axis = 0)

        added_num = points_3d.shape[0]
        offset = self.point_cloud.shape[0] - added_num
        ranges = np.arange(added_num) + offset

        image_data1.refs[pair_data.image_idx1] = ranges
        image_data2.refs[pair_data.image_idx2] = ranges

    def render(self, image_name: str, image_names: list[str], cache: dict[ImageData], matches_3d_2d):
        image_data = cache[image_name]
        kp, des = image_data.kp, image_data.des
        for prev_image_name in cache.keys():
            if prev_image_name != image_name:
                prev_image_data = cache[prev_image_name]
                prev_refs = prev_image_data.refs
                matches = [match for match in matches_3d_2d if image_names[match.imgIdx] == prev_image_name
                           and prev_refs[match.trainIdx] < 0]
                print("New points " + str(len(matches)))
                if len(matches) > 8:
                    def __get_good_matches(matches):
                        matches = sorted(matches, key = lambda m:m.distance)
                        prev_image_idx = np.array([m.trainIdx for m in matches])
                        image_idx = np.array([m.queryIdx for m in matches])

                        kp_prev_image = (np.array(prev_image_data.kp))[prev_image_idx]
                        kp_image = (np.array(kp))[image_idx]

                        points_prev_image = np.array([kp.pt for kp in kp_prev_image])
                        points_image = np.array([kp.pt for kp in kp_image])

                        return  points_prev_image, points_image, prev_image_idx, image_idx

                    points_prev_image, points_image, prev_image_idx, image_idx = __get_good_matches(matches)

                    F, mask = cv.findFundamentalMat(points_prev_image, points_image, cv.FM_RANSAC, 0.1, 0.99)

                    mask = mask.astype(bool).flatten()

                    pair_data = PairData(points_prev_image[mask], points_image[mask], prev_image_idx[mask], image_idx[mask])

                    self.initial_render(prev_image_data, image_data, pair_data)

    def dry_render(self, image_data1, image_data2, image_points1, image_points2):
        image_points_hom1 = cv.convertPointsToHomogeneous(image_points1)[:, 0, :]
        image_points_hom2 = cv.convertPointsToHomogeneous(image_points2)[:, 0, :]

        image_points_norm1 = (self.K_Inv.dot(image_points_hom1.T)).T
        image_points_norm2 = (self.K_Inv.dot(image_points_hom2.T)).T

        image_points_norm1 = cv.convertPointsFromHomogeneous(image_points_norm1)[:, 0, :]
        image_points_norm2 = cv.convertPointsFromHomogeneous(image_points_norm2)[:, 0, :]

        points_4d = cv.triangulatePoints(
            np.hstack((image_data1.R, image_data1.t)),
            np.hstack((image_data2.R, image_data2.t)),
            image_points_norm1.T,
            image_points_norm2.T
        )

        points_3d = cv.convertPointsFromHomogeneous(points_4d.T)[:, 0, :]
        self.point_cloud = np.concatenate((self.point_cloud, points_3d), axis=0)
        print("err1 mini:" + str(error_mini(self.K, image_data1.R, image_data1.t, points_3d, image_points1)))
        print("err2 mini:" + str(error_mini(self.K, image_data2.R, image_data2.t, points_3d, image_points2)))

