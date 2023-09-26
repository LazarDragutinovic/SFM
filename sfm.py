import os
import numpy as np

from image_data import ImageData
from features_manipulator import FeaturesManipulator
from pose_estimator import PoseEstimator
from point_cloud_generator import  PointCloudGenerator

from utility_fucntions import  draw_to_ply, compute_reprojection_error

class SFM:
    image_names:list[str]
    K = None
    def __init__(self, image_dir: str, camera_conf_path:str):
        self.image_names = [os.path.join(image_dir, x) for x in os.listdir(image_dir)]
        if len(self.image_names) < 2:
            raise Exception('Not enough images.')

        with open(camera_conf_path, 'r') as camera_conf:
            #fx;fy;u;v;
            params = list(map(lambda p: float(p),camera_conf.readline().split(';')))
            self.K = np.array([
                [params[0], 0, params[2]],
                [0,  params[1], params[3]],
                [0,   0,        1]
            ])

        self.point_cloud_generator = PointCloudGenerator(self.K)

    def Run(self):
        image_name1 , image_name2 = self.image_names[0], self.image_names[1]

        cache:dict[ImageData] = dict()
        features_manipulator = FeaturesManipulator()

        pose_estimator = PoseEstimator(cache = cache, features_manipulator = features_manipulator, K = self.K)
        R, t, pair_data = pose_estimator.initial_run(image_name1, image_name2)

        self.point_cloud_generator.initial_render(cache[image_name1], cache[image_name2], pair_data)
        error1 = compute_reprojection_error(image_name1, cache, self.point_cloud_generator.point_cloud, self.K)
        error2 = compute_reprojection_error(image_name2, cache, self.point_cloud_generator.point_cloud, self.K)
        print("error " + image_name1 + ": " + str(error1))
        print("error " + image_name2 + ": " + str(error2))

        for new_name in self.image_names[2:]:
            R, t, matches_3d_2d = pose_estimator.run(new_name, self.point_cloud_generator.point_cloud, self.image_names)
            self.point_cloud_generator.render(new_name, self.image_names, cache, matches_3d_2d)
            err = compute_reprojection_error(new_name, cache, self.point_cloud_generator.point_cloud, self.K)
            print("error " + new_name + ": " + str(err))

        mask = self.point_cloud_generator.optimised_point_cloud_mask()

        draw_to_ply("pc.ply", cache, self.point_cloud_generator.point_cloud * 10, mask)