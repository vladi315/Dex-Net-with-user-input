'''
This file automates the virtual evaluation of Vladislav Klass' master thesis.
'''
from pyDOE import *

from fmp_synthetic_data_preprocessing import findAllFile
from fmp_thesis_evaluation_policy_klass import main

if __name__ == '__main__':
    # setup schedule

    # for part_dir in parts:
    #     view_list = sorted(findAllFile(part_dir, 'depth_raw.png'))
    part_dir = "/home/vladislav/gqcnn/data/virtual_evaluation/Housing"
    view_list = sorted(findAllFile(part_dir, 'depth_raw.png'))
    for view in view_list:
        model_name = "GQCNN-2.0"
        camera_intr_filename = "data/calib/basler/basler.intr"
        config_filename = "cfg/examples/replication/dex-net_2.0.yaml"
        camera_pose_path = "data/virtual_evaluation/Housing/poses/0_pose.txt"
        user_input_3d_folder = "data/virtual_evaluation/Housing/tracepen_points"
        user_input_fusion_method = "quadratic_distance_scaling"
        user_input_weight = "high"
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
            user_input_weight = user_input_weight
            )
    




