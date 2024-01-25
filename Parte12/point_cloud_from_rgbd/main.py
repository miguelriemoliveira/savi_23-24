#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA


from copy import deepcopy
import math
from matplotlib import cm
import numpy as np
import open3d as o3d
from more_itertools import locate

view = {"class_name": "ViewTrajectory",
        "interval": 29,
        "is_loop": False,
        "trajectory":
        [
            {
                "boundingbox_max": [6.5291471481323242, 34.024543762207031, 11.225864410400391],
                "boundingbox_min": [-39.714397430419922, -16.512752532958984, -1.9472264051437378],
                "field_of_view": 60.0,
                "front": [0.48005911651460004, -0.71212541184952816, 0.51227008740444901],
                "lookat": [-10.601035566791843, -2.1468729890773046, 0.097372916445466612],
                "up": [-0.28743522255406545, 0.4240317338845464, 0.85882366146617084],
                "zoom": 0.3412
            }
        ],
        "version_major": 1,
        "version_minor": 0
        }


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------

    filename_rgb_image = '/home/mike/Desktop/Rita/rgbd-scenes-v2/imgs/scene_02/00000-color.png'
    filename_depth_image = '/home/mike/Desktop/Rita/rgbd-scenes-v2/imgs/scene_02/00000-depth.png'
    color_raw = o3d.io.read_image(filename_rgb_image)
    depth_raw = o3d.io.read_image(filename_depth_image)
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw, depth_raw, depth_scale=6000, convert_rgb_to_intensity=False)

    print(type(color_raw))
    # K = [fx 0 cx]
    #     [0 fy cy]
    #     [0 0  1]

    fx = 570
    fy = 570
    cx = 320
    cy = 240
    width = 640
    height = 480
    K = o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    print(K)

    # point_cloud = o3d.geometry.PointCloud.create_from_color_and_depth(rgbd_image, K)
    point_cloud = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, K)

    print(len(point_cloud.points))
    # --------------------------------------
    # Visualization
    # --------------------------------------

    entities = [point_cloud]
    # entities.append(point_cloud_horizontal)

    # entities = [point_cloud_original]
    # entities.append(point_cloud_horizontal)

    o3d.visualization.draw_geometries(entities,
                                      zoom=0.3412,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'],
                                      point_show_normal=False)

    # --------------------------------------
    # Termination
    # --------------------------------------


if __name__ == "__main__":
    main()
