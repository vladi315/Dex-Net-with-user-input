from __future__ import print_function

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from visualization import Visualizer2D as vis


def projection(camera_pose, points_3d, camera_matrix, camera_height, camera_width):
    T = np.linalg.inv(camera_pose)
    tvec =np.array(T[0:3, 3])
    rvec, _ = cv2.Rodrigues(T[:3,:3]) 
    point2d = cv2.projectPoints(np.array(points_3d), rvec, tvec, camera_matrix, None)
    points_ref = []
    for p in point2d[0].squeeze():
        i, j = [round(p[1]), round(p[0])]
        print(i,j)
        if i < camera_height and i >= 0 and j < camera_width and j >= 0:
            points_ref.append([j,i])
    
    return np.array(points_ref)

def generate_mask_from_3d_user_input_pos(camera_intrinsics, projected_2d_user_input, img_path, user_input_weight = "low", depth = 0.7):
    '''
    projected_2d_user_input: user input points projected from world to image coordinates
    img_path: path of the image to be masked
    user_input_weight: Controls how strong the effect of the user input is on the final grasp pose prediction. Higher values lead to final grasp predictions closer to the user input location. Choose between 'low', 'medium' and 'high'.")
    depth: depth of user input in camera coordinate system [m] 
    '''
    image_width = camera_intrinsics.width
    image_height = camera_intrinsics.height
    camera_intrinsics_matrix = camera_intrinsics.K

    # calculate radius around user input position that will be masked [m]
    if user_input_weight == "very high":
        mask_radius_in_m = 0.005
    elif user_input_weight == "high":
        mask_radius_in_m = 0.015
    elif user_input_weight == "medium":
        mask_radius_in_m = 0.045
    elif user_input_weight == "low":
        mask_radius_in_m = 0.135

    
    # transform from meters to pixels
    mask_radius_in_pixels = int(mask_radius_in_m * camera_intrinsics_matrix[1,1] / depth)
    mask_image = np.zeros((image_height, image_width, 3), np.uint8)
    for i in range(len(projected_2d_user_input)):
        cv2.circle(mask_image, projected_2d_user_input[i], mask_radius_in_pixels, (255, 255, 255), -1)

    # save mask
    file_name = img_path.strip('.png') + '_user_input_mask.png'
    cv2.imwrite(file_name, mask_image)
    return file_name # TODO: refactor, to return data instead of the saved file name

def project_user_input_to_image(camera_pose_path, pen_folder,  K, H, W):
    pose = np.loadtxt(camera_pose_path)
    pen_files = sorted(glob.glob(os.path.join(pen_folder, "*")))
    pen_points = [np.loadtxt(f) for f in pen_files]
    tracepen_point_2d = projection(pose, pen_points, K, H, W)
    print("tracepen points: ", tracepen_point_2d)
    return tracepen_point_2d

def visualize_tracepen_projection_rgb(img_path, tracepen_point_2d):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.scatter(tracepen_point_2d[:,0], tracepen_point_2d[:,1], c="blue")
    plt.show()
