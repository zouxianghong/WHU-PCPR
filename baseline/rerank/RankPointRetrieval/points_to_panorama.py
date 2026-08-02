#################### Refer to Rank-PointRetrieval ####################

# from: https://www.it610.com/article/1279849167459467264.htm

import numpy as np


# ==============================================================================
#                                                                   SCALE_TO_255
# ==============================================================================
def scale_to_255(a, _min, _max, dtype=np.uint8):
    """ Scales an array of values from specified min, max range to 0-255
        Optionally specify the data type of the output (default is uint8)
    """
    return (((a - _min) / float(_max - _min)) * 255).astype(dtype)


# ==============================================================================
#                                                        POINT_CLOUD_TO_PANORAMA
# ==============================================================================
def point_cloud_to_panorama(points,
                            v_res=0.42,
                            h_res=0.35,
                            v_fov=(-24.9, 2.0),
                            d_range=(0, 100),
                            y_fudge=3
                            ):
    """ Takes point cloud data as input and creates a 360 degree panoramic
        image, returned as a numpy array.

    Args:
        points: (np array)
            The numpy array containing the point cloud. .
            The shape should be at least Nx3 (allowing for more columns)
            - Where N is the number of points, and
            - each point is specified by at least 3 values (x, y, z)
        v_res: (float)
            vertical angular resolution in degrees. This will influence the
            height of the output image.
        h_res: (float)
            horizontal angular resolution in degrees. This will influence
            the width of the output image.
        v_fov: (tuple of two floats)
            Field of view in degrees (-min_negative_angle, max_positive_angle)
        d_range: (tuple of two floats) (default = (0,100))
            Used for clipping distance values to be within a min and max range.
        y_fudge: (float)
            A hacky fudge factor to use if the theoretical calculations of
            vertical image height do not match the actual data.
    Returns:
        A numpy array representing a 360 degree panoramic image of the point
        cloud.
    """
    # Projecting to 2D
    x_points = points[:, 0]
    y_points = points[:, 1]
    z_points = points[:, 2]
    d_points = np.sqrt(x_points ** 2 + y_points ** 2)  # map distance relative to origin
    # d_points = np.sqrt(x_points**2 + y_points**2 + z_points**2)  # abs distance

    # We use map distance, because otherwise it would not project onto a cylinder,
    # instead, it would map onto a segment of slice of a sphere.

    # RESOLUTION AND FIELD OF VIEW SETTINGS
    v_fov_total = -v_fov[0] + v_fov[1]

    # CONVERT TO RADIANS
    v_res_rad = v_res * (np.pi / 180)
    h_res_rad = h_res * (np.pi / 180)

    # MAPPING TO CYLINDER
    x_img = np.arctan2(y_points, x_points) / h_res_rad
    y_img = -(np.arctan2(z_points, d_points) / v_res_rad)

    # THEORETICAL MAX HEIGHT FOR IMAGE
    d_plane = (v_fov_total / v_res) / (v_fov_total * (np.pi / 180))
    h_below = d_plane * np.tan(-v_fov[0] * (np.pi / 180))
    h_above = d_plane * np.tan(v_fov[1] * (np.pi / 180))
    y_max = int(np.ceil(h_below + h_above + y_fudge))

    # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
    x_min = -360.0 / h_res / 2
    x_img = np.trunc(-x_img - x_min).astype(np.int32)
    x_max = int(np.ceil(360.0 / h_res))

    y_min = -((v_fov[1] / v_res) + y_fudge)
    y_img = np.trunc(y_img - y_min).astype(np.int32)

    # DELETE ALL POINTS OUT OF RANGE
    mask = y_img < y_max + 1
    mask2 = y_img >= 0
    mask = mask & mask2

    # CLIP DISTANCES
    # d_points = np.clip(d_points, a_min=d_range[0], a_max=d_range[1])

    # CONVERT TO IMAGE ARRAY
    # img = np.zeros([y_max + 1, x_max + 1], dtype=np.uint8)
    # img[y_img, x_img] = scale_to_255(d_points, _min=d_range[0], _max=d_range[1])

    # SAVE TO DEPTH ARRAY
    #  ONLY SMALLER DEPTH WILL BE SAVED
    depth = np.zeros([y_max + 1, x_max + 1], dtype=np.float)
    for i in range(len(d_points)):
        if not mask[i]:
            continue
        v = depth[y_img[i], x_img[i]]
        if v == 0 or v > d_points[i]:
            depth[y_img[i], x_img[i]] = d_points[i]
    # depth[y_img[mask], x_img[mask]] = d_points[mask]

    return depth


# ==============================================================================
#                                                POINT_CLOUD_TO_PANORAMA(SPHERE)
# ==============================================================================
def point_cloud_to_panorama_sphere(points, cx=0.0, cy=0.0, cz=0.0, h=100, z_zoom=1.0):
    # SPHERE could only be projected to rect with width:height=2:1
    w = h * 2

    # Projecting to 2D
    x_points = points[:, 0] - cx
    y_points = points[:, 1] - cy
    z_points = points[:, 2] * z_zoom - cz
    d_points = np.sqrt(x_points**2 + y_points**2 + z_points**2)  # abs distance

    # MAPPING TO SPHERE
    theta_points = np.arccos(z_points / d_points)  # in vertical direction
    pha_points = np.arctan2(y_points, x_points)  # in horizontal direction

    # MAPPING TO IMAGE
    y_img = -h * theta_points / np.pi
    x_img = w * pha_points / (2 * np.pi)

    # CONVERT TO INT
    x_img = np.trunc(x_img).astype(np.int32)
    y_img = np.trunc(y_img).astype(np.int32)

    # SAVE TO DEPTH ARRAY
    depth = np.zeros([h, w], dtype=np.float)
    for i in range(len(d_points)):
        v = depth[y_img[i], x_img[i]]
        if v == 0 or v > d_points[i]:
            depth[y_img[i], x_img[i]] = d_points[i]

    return depth


# ------------------------------------------------------------------
if __name__ == "__main__":
    import cv2
    import trimesh

    mesh_cur = trimesh.load("./DATA/m0_n1_i17/cur.ply")
    pts_cur = mesh_cur.vertices

    mesh_gt = trimesh.load("./DATA/m0_n1_i17/gt1_17.ply")
    pts_gt = mesh_gt.vertices

    _v_res = 0.3
    _h_res = 0.5
    _v_fov = (-10.0, 45.0)
    # img_cur = point_cloud_to_panorama(pts_cur, v_res=_v_res, h_res=_h_res, v_fov=_v_fov, y_fudge=0)
    # img_gt = point_cloud_to_panorama(pts_gt, v_res=_v_res, h_res=_h_res, v_fov=_v_fov, y_fudge=0)

    _z = 0.0
    _h = 200
    img_cur = point_cloud_to_panorama_sphere(pts_cur, cz=_z, h=_h)
    img_gt = point_cloud_to_panorama_sphere(pts_gt, cz=_z, h=_h)

    cv2.imwrite("IMG_cur.png", img_cur * 255)
    cv2.imwrite("IMG_gt.png", img_gt * 255)
    # np.savetxt('mask.txt', mask)

    # img_tmp = cv2.subtract(img_gt, img_cur)
    # cv2.imwrite("IMG_minus.png", img_tmp * 255)