'''
This file automates the virtual evaluation of Vladislav Klass' master thesis.
'''
import pandas as pd
from numpy import NaN
from pyDOE import *

from fmp_synthetic_data_preprocessing import findAllFile, findAllSubdirectories
from fmp_thesis_evaluation_policy_klass import run_dex_net


def run_virtual_experiments(model_name, camera_intr_filename, config_filename, evaluation_dir):

    ''' 
    setup design of experiment (DOE) schedule.

    Parameters that are varied and their range: 
    1. part [all part subfolders within './data/virtual_evaluation']
    2. user input point [all points in '/user_input_points/' subdirectory]

    3. user_input_fusion_method ["masking", "linear_distance_scaling", "quadratic_distance_scaling"]
    4. user_input_weight ["zero", "very low", "low", "medium", "high", "very high"]
    '''

    # find all object directories
    object_list = sorted(findAllSubdirectories(evaluation_dir))

    # setup up evaluation scheme

    evaluation_template = {'object_path':[], 
                            'user_input_point_number':[], 
                            'user input fusion method':[], 
                            'user input weight':[], 
                            'distance_grasp_to_user_input_norm':[], 
                            'grasp_quality':[], 
                            'mean_evaluation_metric':[]}

    evaluation_scheme = pd.DataFrame(data=evaluation_template)

    for _, object_dir in enumerate(object_list):
        # get all camera view points
        view_list = sorted(findAllFile(object_dir, 'depth_raw.png'))
        for view_idx, view in enumerate(view_list[0:1]): # to save computatin time, only use the first view
            # set paths
            camera_pose_path = object_dir + "/poses/0_pose.txt"
            user_input_3d_folder = object_dir + "/user_input_points"

            point_list = sorted(findAllFile(object_dir + "/user_input_points/", 'point.txt'))
            # iterate through all saved user input points 
                
            if "suction" in config_filename:
                # load segmentation mask

                last_slash_index = object_dir[::-1].index("/")
                obect_name = object_dir[-last_slash_index:]

                segmask_filename = evaluation_dir + "/../masks/" + obect_name + "_mask/" + str(view_idx) + ".png"
            else: 
                segmask_filename = None

            for user_input_fusion_method in ["masking"]: # ["masking", "linear_distance_scaling", "quadratic_distance_scaling"]:
                for user_input_weight in ["low"]: # ["low", "medium", "high", "very high"]
                    for point_idx in range(min(len(point_list), 1)): # limit user input points max to save computation time

                        try:
                            mean_evaluation_metric, grasp_quality, distance_grasp_to_user_input_norm = run_dex_net(
                                model_name,
                                depth_ims_dir  = None,
                                depth_im_filename = view, 
                                segmask_filename = segmask_filename, 
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
                            experiment = pd.DataFrame({'object_path': [object_dir], 
                            'user_input_point_number':[point_idx], 
                            'user input fusion method':[user_input_fusion_method], 
                            'user input weight':[user_input_weight], 
                            'distance_grasp_to_user_input_norm':[distance_grasp_to_user_input_norm], 
                            'grasp_quality':[grasp_quality], 
                            'mean_evaluation_metric':[mean_evaluation_metric]})
                        except:
                            # If no valid grasps are found
                            print("No valid grasp")
                            experiment = pd.DataFrame({'object_path': [object_dir], 
                            'user_input_point_number':[point_idx], 
                            'user input fusion method':[user_input_fusion_method], 
                            'user input weight':[user_input_weight], 
                            'distance_grasp_to_user_input_norm':[NaN], 
                            'grasp_quality':[0], 
                            'mean_evaluation_metric':[0]})
                        
                        evaluation_scheme = evaluation_scheme.append(experiment)

    
        # Save data to csv
        evaluation_scheme.to_csv(evaluation_dir + '/virtual_experiments_results_pj.csv', index=False)        

def postprocess_experiment_data(evaluation_dir):
    # df = pd.read_csv(evaluation_dir + '/virtual_experiments_results.csv')
    df = pd.read_csv('11_evaluation_virtual_experments_user_input_fusion.csv')
    
if __name__ == '__main__':
    
    model_name = "GQCNN-4.0-SUCTION" #  GQCNN-4.0-PJ GQCNN-2.0 GQCNN-4.0-SUCTION
    camera_intr_filename = "data/calib/basler/basler.intr"
    config_filename = "cfg/examples/replication/dex-net_4.0_suction.yaml" # dex-net_4.0_pj.yaml dex-net_2.0.yaml dex-net_4.0_suction.yaml
    evaluation_dir = "/home/vladislav/gqcnn/data/virtual_evaluation/13objsSuction/renderings"

    run_virtual_experiments(model_name, camera_intr_filename, config_filename, evaluation_dir)

    # postprocess_experiment_data(evaluation_dir)



