import numpy as np
import cv2 as cv

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

def draw_to_ply(file_name:str, cache: dict[ImageData], point_cloud, mask):
    colors = np.zeros_like(point_cloud)
    for k in cache.keys():
        image_data = cache[k]
        kp, des, refs = image_data.kp, image_data.des, image_data.refs
        kp = np.array(kp)[refs >= 0]
        image_points = np.array([_kp.pt for _kp in kp])

        image_mat = cv.imread(k)[:,:,::-1]
        colors[refs[refs >= 0].astype(int)] = image_mat[image_points[:,1].astype(int), image_points[:,0].astype(int)]

    pts2ply(point_cloud[mask], colors[mask], file_name)


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