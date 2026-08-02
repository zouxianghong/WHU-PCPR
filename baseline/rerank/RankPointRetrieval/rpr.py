#################### Refer to Rank-PointRetrieval ####################

import open3d
import numpy as np
import copy
from sklearn.neighbors import KDTree
import time

from baseline.rerank.RankPointRetrieval.points_to_panorama import point_cloud_to_panorama_sphere

# type for evaluating registration results
type_reg = 'vc'
# which type of keypoints to use, and how much
num_keypts = None
type_keypts = None
# if true, output logs when evaluating registration
LOG = False
# if true, output time using
LOG_TIME = False
# for debugging
DEBUG = False

WITH_NOISES = True
SIGMA = 0.01


def init(keypoints=32, keytype='rand',
         reg_type = 'vc', with_noise=False, sigma=0.01, log_time=False):

    global type_reg, num_keypts, type_keypts
    type_reg = reg_type
    num_keypts = keypoints
    type_keypts = keytype

    global data_fpfh

    global WITH_NOISES
    WITH_NOISES = with_noise

    global SIGMA
    SIGMA = sigma

    global LOG_TIME
    LOG_TIME = log_time


def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    # print('sigma', sigma)
    # if sigma > 0.01:
    #     clip = clip/0.01*0.05
    #     print('clip', clip)
    clip = 1.0
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data


# ------------------------------------------------------------------
# pcd is a np.array, output the open3d's geometry
def convert_to_open3d_geometry(kpt, feat):
    if WITH_NOISES:
        kpt_expand = np.expand_dims(kpt, axis=0)
        kpt = jitter_point_cloud(kpt_expand, SIGMA).squeeze()

    s = make_open3d_point_cloud(kpt)
    s_f = make_open3d_feature(feat, dim=feat.shape[1], npts=feat.shape[0])
    return s, s_f


# ------------------------------------------------------------------
# using fast global registration
def run_fgr(s1, s2, s1_f, s2_f, distance_threshold=0.5):
    start = time.time()
    reg = open3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        s2, s1, s2_f, s1_f,
        open3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold)
        )
    if LOG_TIME:
        print("FGR time %.3f sec" % (time.time() - start))
    # start = time.time()
    # add a ICP registration for a more accurate result
    # reg = open3d.pipelines.registration.registration_icp(s2, s1, distance_threshold, reg.transformation,
    #    open3d.pipelines.registration.TransformationEstimationPointToPoint(),
    #    open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration = 50))
    # print("ICP time %.3f sec" % (time.time() - start))
    # trans_init = np.asarray([[1., 0., 0., 0.],
    #                          [0., 1., 0., 0.],
    #                          [0., 0., 1., 0.],
    #                          [0., 0., 0., 1.]])
    return reg


# ------------------------------------------------------------------
def draw_registration_result(s1, s2, transformation, name):
    s1_temp = copy.deepcopy(s1)
    s2_temp = copy.deepcopy(s2)
    s1_temp.paint_uniform_color([1, 0.706, 0])
    s2_temp.paint_uniform_color([0, 0.651, 0.929])
    s2_temp.transform(transformation)
    open3d.io.write_point_cloud(name, s1_temp + s2_temp)


# ------------------------------------------------------------------
# pc1 : the origin pts of the source
# pc2_new : the target pts after reg.transformation
# result : the dict to store results (pointer)
def evaluate_with_rigid_match(pc1, pc2_new, result):
    tree1 = KDTree(pc1)
    dist2to1, ind2to1 = tree1.query(pc2_new)
    tree2 = KDTree(pc2_new)
    dist1to2, ind1to2 = tree2.query(pc1)

    # for ith point in pc1, the corresponding point in pc2 is ind1to2[i]
    # for every point, add a flag to set if it has a rigid registration
    reg1to2_score = np.ndarray((pc1.shape[0], 1), dtype=np.float32)
    for i in range(0, ind1to2.shape[0]):
        reg1to2_score[i] = 0

    # check if the point has a rigid corresponding match
    max_match_d = 0.05  # if the distance is bigger than this value, it's not a good match
    # fail_match_d = 0.30  # if the distance is too big, than it's a failure match
    for i in range(0, ind1to2.shape[0]):
        reg_i = ind1to2[i]
        # if dist1to2[i] > fail_match_d:
        #     reg1to2_score[i] = 0
        #     continue
        if ind2to1[reg_i] == i:
            if dist1to2[i] <= max_match_d:
                reg1to2_score[i] = 1.0
                continue
            # reg1to2_score[i] = (fail_match_d - dist1to2[i])
        # else:
            # reg1to2_score[i] = -0.02

    v_match = sum(reg1to2_score)[0]
    r_match = v_match / reg1to2_score.shape[0]

    result['or'] = r_match


# ------------------------------------------------------------------
# pc1 : the origin pts of the source
# pc2_new : the target pts after reg.transformation
# result : the dict to store results (pointer)
def evaluate_with_panorama(pc1, pc2_new, result):
    cz = 0.0
    z_zoom = 1.0
    h = 90

    depth1 = point_cloud_to_panorama_sphere(pc1, cz=cz, z_zoom=z_zoom, h=h)
    depth2 = point_cloud_to_panorama_sphere(pc2_new, cz=cz, z_zoom=z_zoom, h=h)

    # depth1 = filter_depth_with_filling(depth1)
    # depth2 = filter_depth_with_filling(depth2)

    mask1 = depth1 > 0
    mask2 = depth2 > 0
    mask = mask1 & mask2  # mask for pixels with depth
    n = np.sum(mask1 | mask2)

    dis = depth2 - depth1
    dis = np.abs(dis)

    occ = dis > 0.12
    occ = occ & mask
    lap = dis <= 0.12
    lap = lap & mask

    result['vc_occ'] = np.sum(occ)
    result['vc_lap'] = np.sum(lap)

    result['vc'] = (result['vc_lap'] - result['vc_occ'] * 0.5) / n


def filter_depth_with_filling(depth_img):
    d_fill = 5
    depth_new = np.zeros((depth_img.shape[0], depth_img.shape[1]), dtype=np.float)
    depth_c = np.zeros((depth_img.shape[0], depth_img.shape[1]), dtype=np.float)
    # find pixel that has depth
    ys, xs = np.where(depth_img > 0)
    for i, y in enumerate(ys):
        x = xs[i]
        depth_cur = depth_img[y, x]

        # using the depth to do the filling in 3*3 area
        for y_d in range(d_fill):
            y_ = y + y_d - d_fill // 2
            if y_ < 0 or y_ >= depth_new.shape[0]:
                continue
            for x_d in range(d_fill):
                x_ = x + x_d - d_fill // 2
                if x_ < 0 or x_ >= depth_new.shape[1]:
                    continue
                depth_new[y_, x_] += depth_cur
                depth_c[y_, x_] += 1

    depth_c[depth_c == 0] = 1

    depth_new = depth_new * 1.0 / depth_c
    return depth_new


# ------------------------------------------------------------------
# files_data = [{'pc': n x 3, 'fpfh': 33 x n}]
def evaluate_matchs(a_data, k_datas, distance_threshold=0.5, log=False):
    reg_scores = []

    # anchor
    kpt1, feat1 = convert_to_open3d_geometry(a_data['l_kpt'], a_data['l_desc'])
    # register one by one
    for index in range(len(k_datas)):
        result = {}
        k_data = k_datas[index]
        pc2 = make_open3d_point_cloud(k_data['pc'])
        kpt2, feat2 = convert_to_open3d_geometry(k_data['l_kpt'], k_data['l_desc'])

        # do registration
        if np.sum(feat1.data) < 1.e-10 or np.sum(feat2.data) < 1.e-10:
            result['reg_score'] = -1.0
        else:
            reg = run_fgr(kpt1, kpt2, feat1, feat2, distance_threshold)
            result['trans'] = reg.transformation

            # vis reg result
            if log:
                draw_registration_result(kpt1, kpt2, reg.transformation, '/home/ericxhzou/Code/PCGL-Benchmark/exp/rerank/reg.pcd')

            # apply registration's result
            s2_new = copy.deepcopy(pc2)
            s2_new.transform(reg.transformation)
            pc2_new = np.asarray(s2_new.points)

            # check the registration result
            # however, no results for fgr
            cor_set = np.asarray(reg.correspondence_set)
            result['fgr_setc'] = cor_set.shape[0]
            result['fgr_fitness'] = reg.fitness
            result['fgr_rmse'] = reg.inlier_rmse

            # using custom standard
            kpt1_cp = copy.deepcopy(kpt1)
            kpt1_cp = np.asarray(kpt1_cp.points)
            if type_reg == 'or':
                start = time.time()
                evaluate_with_rigid_match(kpt1_cp, pc2_new, result)
                if LOG_TIME:
                    print("rigid match time %.3f sec" % (time.time() - start))
                result['reg_score'] = result['or']
            if type_reg == 'vc':
                start = time.time()
                evaluate_with_panorama(kpt1_cp, pc2_new, result)
                if LOG_TIME:
                    print("panorama time %.3f sec" % (time.time() - start))
                result['reg_score'] = result['vc']
        reg_scores.append(result['reg_score'])
    return reg_scores


def calc_fpfh(pc, radius_normal=0.05, radius_feature=0.05):
    """ Return: N x d np array """
    s = make_open3d_point_cloud(pc)
    s.estimate_normals(open3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    fpfh = open3d.pipelines.registration.compute_fpfh_feature(
        s, open3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return fpfh


########################################See Spectral GV ########################################
def make_open3d_feature(data, dim, npts):
    feature = open3d.pipelines.registration.Feature()
    feature.resize(dim, npts)
    if not isinstance(data, np.ndarray):
        feature.data = data.cpu().numpy().astype('d').transpose()
    else:
        feature.data = data.astype('d').transpose()
    return feature


def make_open3d_point_cloud(xyz, color=None):
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(xyz)
    if color is not None:
        pcd.colors = open3d.utility.Vector3dVector(color)
    return pcd


def get_ransac_result(feat1, feat2, kp1, kp2, ransac_dist_th=0.5, ransac_max_it=10000, vis=False):
    feature_dim = feat1.shape[1]
    pcd_feat1 = make_open3d_feature(feat1, feature_dim, feat1.shape[0])
    pcd_feat2 = make_open3d_feature(feat2, feature_dim, feat2.shape[0])
    pcd_coord1 = make_open3d_point_cloud(kp1)
    pcd_coord2 = make_open3d_point_cloud(kp2)

    # ransac based eval
    ransac_result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        pcd_coord1, pcd_coord2, pcd_feat1, pcd_feat2,
        mutual_filter=True,
        max_correspondence_distance=ransac_dist_th,
        estimation_method=open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=3,
        checkers=[open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.8),
                  open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(ransac_dist_th)],
        criteria=open3d.pipelines.registration.RANSACConvergenceCriteria(ransac_max_it, 0.999))

    # For Debug: vis reg result
    if vis:
        pcd_coord1.paint_uniform_color([1,0,0])
        pcd_coord2.paint_uniform_color([0,1,0])
        open3d.visualization.draw([pcd_coord1.transform(ransac_result.transformation), pcd_coord2])
    return ransac_result
########################################See Spectral GV ########################################