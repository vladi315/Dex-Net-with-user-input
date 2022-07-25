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

def generate_mask_from_tracepen_pos(H, W, points_2d, img_path, mask_radius, depth):
    '''
    mask radius: radius around tp position that will be masked [m]
    depth: depth of tracepen point in [m] 
    '''
    mask_radius_in_image_plane = int(mask_radius * K[1,1] / depth)
    mask_image = np.zeros((H,W,3), np.uint8)
    for i in range(len(points_2d)):
        cv2.circle(mask_image, points_2d[i], mask_radius_in_image_plane, (255, 255, 255), -1)
    plt.imshow(mask_image)
    plt.show()

    # save mask
    file_name = img_path.strip('.png') + '_tracepen_mask.png'
    cv2.imwrite(file_name, mask_image)

def project_tracepen_points_to_image(pose_path, pen_folder,  K, H, W):
    pose = np.loadtxt(pose_path)
    pen_files = sorted(glob.glob(os.path.join(pen_folder, "*")))
    pen_points = [np.loadtxt(f) for f in pen_files]
    tracepen_point_2d = projection(pose, pen_points, K, H, W)
    print("tracepen points: ", tracepen_point_2d)
    return tracepen_point_2d

def visualize_tracepen_projection_rgb(img_path, tracepen_point_2d):
    img = cv2.cvtColor(cv2.imread(img_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.scatter(tracepen_point_2d[:,0], tracepen_point_2d[:,1], c="red")
    plt.show()

if __name__ == '__main__':
    # Realsense 435i intrinsics
    K = np.array([907.103516, 0.0,  649.697702, 0.0, 907.77832, 383.828184, 0.0, 0.0, 1.0]).reshape(3,3)
    W = 1280
    H = 720

    img_path = "/home/vladislav/Downloads/housing_22_06/data/0000_image.png"
    pose_path = "/home/vladislav/Downloads/housing_22_06/data/0000_pose.txt"
    pen_folder = "/home/vladislav/Downloads/housing_22_06/points/data"
    tracepen_point_2d = project_tracepen_points_to_image(pose_path, pen_folder,  K, H, W)
    visualize_tracepen_projection_rgb(img_path, tracepen_point_2d)

    # generate mask around tracepen points
    mask_radius = 0.03
    depth = 0.7 # TODO: refine this by calculated relative distance camera to tracepen point
    generate_mask_from_tracepen_pos(H, W, tracepen_point_2d, img_path, mask_radius, depth)
