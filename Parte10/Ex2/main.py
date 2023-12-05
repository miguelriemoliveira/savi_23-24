#!/usr/bin/env python3

# -----------------------------------------------------------------
# Project: PSR 2022-2023
# Author: Miguel Riem Oliveira
# Inspired in:
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
# -----------------------------------------------------------------

from copy import deepcopy
import math

import open3d as o3d
import numpy as np
from matplotlib import cm
from more_itertools import locate

view = {
    "class_name": "ViewTrajectory",
    "interval": 29,
    "is_loop": False,
    "trajectory":
        [
            {
                "boundingbox_max": [2.6540005122611348, 2.3321821423160629, 0.85104994623420782],
                "boundingbox_min": [-2.5261458770339673, -2.1656718060235378, -0.55877501755379944],
                "field_of_view": 60.0,
                "front": [0.75672239933786944, 0.34169632162348007, 0.55732830013316348],
                "lookat": [0.046395260625899069, 0.011783639768603466, -0.10144691776517496],
                "up": [-0.50476400916821107, -0.2363660920597864, 0.83026764695055955],
                "zoom": 0.30119999999999997
            }
        ],
    "version_major": 1,
    "version_minor": 0
}


def main():

    # -----------------------------------------------------------------
    # Initialization
    # -----------------------------------------------------------------
    pcd_original = o3d.io.read_point_cloud('../data/scene.ply')

    # -----------------------------------------------------------------
    # Execution
    # -----------------------------------------------------------------

    # Downsample using voxel grid ------------------------------------
    pcd_downsampled = pcd_original.voxel_down_sample(voxel_size=0.02)
    # pcd_downsampled.paint_uniform_color([1,0,0])

    # estimate normals
    pcd_downsampled.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd_downsampled.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

    # Create transformation T1 only with rotation
    T1 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T1[3, 3] = 1

    # Add null rotation
    R = pcd_downsampled.get_rotation_matrix_from_xyz((110*math.pi/180, 0, 40*math.pi/180))
    T1[0:3, 0:3] = R
    # T[0:3, 0] = [1, 0, 0]  # add n vector
    # T[0:3, 1] = [0, 1, 0]  # add s vector
    # T[0:3, 2] = [0, 0, 1]  # add a vector

    # Add a translation
    T1[0:3, 3] = [0, 0, 0]
    print('T1=\n' + str(T1))

    # Create transformation T2 only with translation
    T2 = np.zeros((4, 4), dtype=float)

    # Add homogeneous coordinates
    T2[3, 3] = 1

    # Add null rotation
    T2[0:3, 0] = [1, 0, 0]  # add n vector
    T2[0:3, 1] = [0, 1, 0]  # add s vector
    T2[0:3, 2] = [0, 0, 1]  # add a vector

    # Add a translation
    T2[0:3, 3] = [0.8, 1, -0.4]
    print('T2=\n' + str(T2))

    T = np.dot(T1, T2)
    print('T=\n' + str(T))

    # Create table ref system and apply transformation to it
    frame_table = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.2, origin=np.array([0., 0., 0.]))

    frame_table = frame_table.transform(T)

    pcd_downsampled = pcd_downsampled.transform(np.linalg.inv(T))

    # Visualization ----------------------
    pcds_to_draw = [pcd_downsampled]

    frame_world = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

    entities = []
    entities.append(frame_world)
    # entities.append(frame_table)
    entities.extend(pcds_to_draw)
    o3d.visualization.draw_geometries(entities,
                                      zoom=0.3412,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'], point_show_normal=False)

    # -----------------------------------------------------------------
    # Termination
    # -----------------------------------------------------------------


if __name__ == "__main__":
    main()
