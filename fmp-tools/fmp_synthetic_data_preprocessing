import json
import os

import numpy as np
from cv2 import transform


def findAllFile(base, suffix):
    '''
    :param base: the folder path
    :return: file list in the folder 
    '''
    file_list=[]
    for root, ds, fs in os.walk(base):
        fs.sort()
        for f in fs:
            if f.endswith(suffix):  
                file_list.append(os.path.join(root, f))
    return file_list 

def sample_points(point_number = 10):
    points = []
    for point_idx in range(point_number):
        mean = [0.0, 0.0, 0.005]
        standard_deviations = [0.01, 0.01, 0.005]
        points.append( np.random.normal(mean, standard_deviations))
    return points

def save_points(points, points_dir):
    for point_idx, point in enumerate(points):
        os.makedirs(points_dir, exist_ok=True)
        point_path = points_dir + str(point_idx) + "_point.txt"
        np.savetxt(point_path , point)
    return 

def transform_camera_poses_nerf_to_dexnet(camera_pose_path):
    with open(camera_pose_path, 'r') as f:
        data = json.load(f)
    transform_matrices = []
    for pose_idx, pose in enumerate(data):
        transform_matrix = pose['transform_matrix']
        transform_matrices.append(np.array([transform_matrix[0], transform_matrix[1], transform_matrix[2], transform_matrix[3]]))
    return transform_matrices

def save_camera_poses(transform_matrices, poses_dir):
    for matrix_idx, transform_matrix in enumerate(transform_matrices):
        os.makedirs(poses_dir, exist_ok=True)
        new_file_name = poses_dir.rstrip(".json") + str(matrix_idx) + "_pose" + ".txt"
        np.savetxt(new_file_name, transform_matrix)

if __name__ == '__main__':
    ''' 
    This script converts saved camera poses from the NeRF to the Dex-Net format.
    '''
    root_path = './data/virtual_evaluation/'
    file_list = sorted(findAllFile(root_path, 'transforms_.json'))
    print(file_list)
    for camera_pose_path in file_list:
        # get parent directory 
        last_index = camera_pose_path[::-1].index("/")
        parent_dir = camera_pose_path[:-last_index]
        
        # transform from nerf output format to dexnet input format
        transform_matrices = transform_camera_poses_nerf_to_dexnet(camera_pose_path)
        poses_dir = parent_dir + "poses/"
        save_camera_poses(transform_matrices, poses_dir)

        # fake and save tracepen points
        point_number = 10
        points = sample_points(point_number)
        points_dir = parent_dir + "tracepen_points/"
        save_points(points, points_dir)

