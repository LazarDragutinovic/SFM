import numpy as np
import cv2 as cv

def get_camera_frame(points_4d, R, t):
    return (R.dot(points_4d.T) + t).T

def __find_min_and_max_depth__(points_4d):
    return min(points_4d[2]), max(points_4d[2])


def __choose_eq_distances__(points_4d):
    min_depth, max_depth = __find_min_and_max_depth__(points_4d)

    depths = np.linspace(min_depth, max_depth, 1200, dtype=float)
    return depths


def __get_projection_mtx__(projection, n, d):
    bottom = np.hstack((n, d))
    p = np.vstack((projection, bottom))
    return p


def find_homography(p1, p2, points_4d, intrinsic_matrix):
    depths = __choose_eq_distances__(points_4d)
    n = np.array([0, 0, -1])


    homography = []

    for i in depths:
        projection_2 = __get_projection_mtx__(p2, n, i)
        projection_1 = __get_projection_mtx__(p1, n, i)
        m = np.dot(projection_1, np.linalg.inv(projection_2))
        h = m[0:3, 0:3]

        homography.append(h)

    return homography, depths


def get_warped_images(homography, image2):
    images = []
    for i, h in enumerate(homography):
        img = cv.warpPerspective(image2, h, None)
        images.append(img)

    return images


def run_abs_diff_and_block_filter(image1, image2, projection_1, projection_2, points_4d, intrinsic_matrix):
    homography, depths = find_homography(projection_1, projection_2, points_4d, intrinsic_matrix)

    warped_images = get_warped_images(homography, image2)

    diff = []

    for image in warped_images:
        diff_image = cv.blur(cv.absdiff(image, image1), (40, 40))

        diff.append(diff_image)

    return diff, depths

class PlaneSweep:
    def compute_depth(self ,image_name1, image_name2, cache:dict, point_cloud, K):
        image_data1 = cache[image_name1]
        image_data2 = cache[image_name2]
        projection_1 = np.matmul(K, np.hstack((np.eye(3), np.zeros((3, 1)))))
        R1 = image_data1.R
        R2 = image_data2.R
        t1 = image_data1.t
        t2 = image_data2.t
        R1_inv = np.linalg.inv(R1)

        projection_2 = np.matmul(K, np.hstack((R2.dot(R1_inv), t2 - R1_inv.dot(t1))))

        refs1, refs2 = image_data1.refs, image_data2.refs

        common_points = point_cloud[np.intersect1d(refs1[refs1>0].astype(int), refs2[refs2>0].astype(int))]

        common_points = R1.dot(common_points.T) + t1
        common_points = common_points.T

        points_4d = cv.convertPointsToHomogeneous(common_points)[:, 0, :]

        img1Gray = cv.cvtColor(image_data1.mat, cv.COLOR_BGR2GRAY)
        img2Gray = cv.cvtColor(image_data2.mat, cv.COLOR_BGR2GRAY)

        warped_abs_diff, depths = run_abs_diff_and_block_filter(
            img1Gray, img2Gray,
            projection_1, projection_2, points_4d, K
        )

        shp = img1Gray.shape

        image1 = img1Gray.ravel()
        depth_matrix = np.zeros(image1.shape)

        ravel_diff = []

        for image in warped_abs_diff:
            img = image.ravel()
            ravel_diff.append(img)

        mat = np.array(ravel_diff)

        for pixel in range(len(mat[0])):
            minimum = np.argmin(mat[:, pixel])

            depth_matrix[pixel] = depths[minimum]

        depth_matrix = depth_matrix.reshape(shp)
        return  depth_matrix
