#!/usr/bin/env python3

# -----------------------------------------------------------------
# Project: PSR 2022-2023
# Author: Miguel Riem Oliveira
# Inspired in:
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html
# https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
# -----------------------------------------------------------------

from copy import deepcopy

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
                "boundingbox_max": [3.0000000000000004, 3.0000000000000004, 3.83980393409729],
                "boundingbox_min": [-2.5246021747589111, -1.5300980806350708, -1.4928504228591919],
                "field_of_view": 60.0,
                "front": [0.68416710190818031, -0.67831952094493897, -0.2679514959308687],
                "lookat": [0.66304140187417715, 0.065777083788090218, 1.8040776836764005],
                "up": [-0.72422933700365322, -0.67522867403826003, -0.13985029560134044],
                "zoom": 0.43999999999999972
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

    # Visualization ----------------------
    pcds_to_draw = [pcd_downsampled]

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=0.5, origin=np.array([0., 0., 0.]))

    entities = []
    entities.append(frame)
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
