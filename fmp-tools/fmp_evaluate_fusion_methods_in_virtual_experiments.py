'''
This file automates the virtual evaluation of Vladislav Klass' master thesis.
'''
from pyDOE import *

from fmp_synthetic_data_preprocessing import findAllFile, findAllSubdirectories
from fmp_thesis_evaluation_policy_klass import main

if __name__ == '__main__':
    ''' 
    setup design of experiment (DOE) schedule.

    Parameters that are varied and their range: 
    1. part [all part subfolders within './data/virtual_evaluation']
    2. user input point [all points in '/user_input_points/' subdirectory]

    3. user_input_fusion_method ["masking", "linear_distance_scaling", "quadratic_distance_scaling"]
    4. user_input_weight ["zero", "very low", "low", "medium", "high", "very high"]
    '''

    model_name = "GQCNN-4.0-PJ"
    camera_intr_filename = "data/calib/basler/basler.intr"
    config_filename = "cfg/examples/replication/dex-net_4.0_pj.yaml"

    evaluation_dir = "./data/virtual_evaluation/"
    # find all object directories
    object_list = sorted(findAllSubdirectories(evaluation_dir))
    for object_dir in object_list:
        # get all camera view points
        view_list = sorted(findAllFile(object_dir, 'depth_raw.png'))
        for view in view_list[0:1]: # to save computatin time, only use the first view
            # set paths
            camera_pose_path = object_dir + "/poses/0_pose.txt"
            user_input_3d_folder = object_dir + "/user_input_points"

            # set sampled parameters
            user_input_fusion_method = "quadratic_distance_scaling"
            user_input_weight = "high"

            point_list = sorted(findAllFile(object_dir + "/user_input_points/", 'point.txt'))
            # iterate through all saved user input points 
            for point_idx in range(min(len(point_list), 5)): # limit user input points max to save computation time
                mean_evaluation_metric, grasp_quality, distance_grasp_to_user_input_norm = main(
                    model_name,
                    depth_ims_dir  = None,
                    depth_im_filename = view, 
                    segmask_filename = None, 
                    camera_intr_filename = camera_intr_filename, 
                    model_dir = None, 
                    config_filename = config_filename, 
                    fully_conv = None, 
                    camera_pose_path = camera_pose_path, 
                    user_input_3d_folder = user_input_3d_folder, 
                    user_input_fusion_method = user_input_fusion_method, 
                    user_input_weight = user_input_weight,
                    user_input_point_number = point_idx
                    )
                print(mean_evaluation_metric, grasp_quality, distance_grasp_to_user_input_norm)
        




