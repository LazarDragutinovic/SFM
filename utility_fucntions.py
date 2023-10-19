import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

from image_data import ImageData

def pts2ply(pts, colors, filename='out.ply'):
    """Saves an ndarray of 3D coordinates (in meshlab format)"""

    with open(filename, 'w') as f:
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex {}\n'.format(pts.shape[0]))

        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')

        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')

        f.write('end_header\n')

        # pdb.set_trace()
        colors = colors.astype(int)
        for pt, cl in zip(pts, colors):
            f.write('{} {} {} {} {} {}\n'.format(pt[0], pt[1], pt[2],
                                                 cl[0], cl[1], cl[2]))

def draw_to_ply_sparse(file_name:str, cache: dict, point_cloud,K):
    colors = np.zeros((0, 3))
    for k in cache.keys():
        image_data = cache[k]
        refs = image_data.refs
        kps = image_data.kp[image_data.refs>0]
        points_3d = point_cloud[refs[refs>0]]
        points_2d = K.dot(image_data.R.dot(points_3d.T) + image_data.t).T
        points_2d = cv.convertPointsFromHomogeneous(points_2d)[:, 0, :]
        points_2d = points_2d.astype(int)
        for x in range(len(points_2d)):
            if points_2d[x, 0] < 0:
                points_2d[x, 0] = 0

            if points_2d[x, 0] >= image_mat.shape[1]:
                points_2d[x, 0] = image_mat.shape[1] - 1

            if points_2d[x, 1] < 0:
                points_2d[x, 1] = 0

            if points_2d[x, 1] >= image_mat.shape[0]:
                points_2d[x, 1] = image_mat.shape[0] - 1
        colors = np.concatenate((colors, image_mat[points_2d[:, 1], points_2d[:, 0]]), axis=0)
    pts2ply(point_cloud, colors, file_name)

def draw_to_ply(file_name:str, cache: dict, K):
    dense_colors = np.zeros((0, 3))
    pc = np.zeros((0, 3))

    idx = 0
    for k in cache.keys():
        if idx == 2:
            break

        idx += 1
        image_data = cache[k]
        image_mat = image_data.mat
        if image_data.dense_points.any():
            pc = np.concatenate((pc,  image_data.dense_points), axis=0)
            points_2d = K.dot(image_data.R.dot(image_data.dense_points.T) + image_data.t)
            points_2d = cv.convertPointsFromHomogeneous(points_2d.T)[:, 0, :]
            points_2d = points_2d.astype(int)
            for x in range(len(points_2d)):
                if points_2d[x,0] < 0:
                    points_2d[x,0] = 0

                if points_2d[x, 0] >= image_mat.shape[1]:
                    points_2d[x, 0] = image_mat.shape[1] - 1

                if points_2d[x, 1] < 0:
                    points_2d[x, 1] = 0

                if points_2d[x, 1] >= image_mat.shape[0]:
                    points_2d[x, 1] = image_mat.shape[0] - 1

            dense_colors = np.concatenate((dense_colors ,image_mat[points_2d[:,1], points_2d[:,0]][:,::-1]), axis=0)
    rotx = np.array([[1, 0, 0],
                    [0, np.cos(np.pi), -np.sin(np.pi)],
                    [0, np.sin(np.pi), np.cos(np.pi)]])
    pc = rotx.dot(pc.T).T
    pts2ply(pc, dense_colors, file_name)


def compute_reprojection_error(image_name:str, cache:dict[ImageData], point_cloud, K):
    image_data = cache[image_name]
    kp, des, refs = image_data.kp, image_data.des, image_data.refs
    R, t = image_data.R, image_data.t

    points_3d = point_cloud[refs[refs>=0].astype(int)]

    points_2d = K.dot(R.dot(points_3d.T) + t)
    points_2d = cv.convertPointsFromHomogeneous(points_2d.T)[:,0,:]

    points_image_2d = np.array([kp.pt for i , kp in enumerate(kp) if refs[i] >= 0])

    sum = np.sum((points_image_2d - points_2d) ** 2, axis=-1)
    sqrt = np.sqrt(sum)

    err = np.mean(sqrt)

    return  err

def error_mini(K, R, t, points_3d, points_image_2d):
    points_2d = K.dot(R.dot(points_3d.T) + t)
    points_2d = cv.convertPointsFromHomogeneous(points_2d.T)[:, 0, :]

    sum = np.sum((points_image_2d - points_2d) ** 2, axis=-1)
    sqrt = np.sqrt(sum)

    err = np.mean(sqrt)

    return err