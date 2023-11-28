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


class PlaneSegmentation():

    def __init__(self, input_point_cloud):

        self.input_point_cloud = input_point_cloud
        self.a = None
        self.b = None
        self.c = None
        self.d = None

    def findPlane(self, distance_threshold=0.35, ransac_n=3, num_iterations=100):
        plane_model, inlier_idxs = self.input_point_cloud.segment_plane(distance_threshold=distance_threshold,
                                                                        ransac_n=ransac_n,
                                                                        num_iterations=num_iterations)
        self.a, self.b, self.c, self.d = plane_model

        self.inliers = self.input_point_cloud.select_by_index(inlier_idxs, invert=False)
        self.outliers = self.input_point_cloud.select_by_index(inlier_idxs, invert=True)


def main():

    # --------------------------------------
    # Initialization
    # --------------------------------------
    filename = '../Ex4/factory_isolated.pcd'
    print('Loading file ' + filename)
    point_cloud_original = o3d.io.read_point_cloud(filename)

    # --------------------------------------
    # Estimate normals
    # --------------------------------------
    point_cloud_original.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30))
    point_cloud_original.orient_normals_to_align_with_direction(orientation_reference=np.array([0, 0, 1]))

    point_cloud_horizontal = o3d.geometry.PointCloud()

    for point, normal, color in zip(point_cloud_original.points,
                                    point_cloud_original.normals,
                                    point_cloud_original.colors):

        # Compute angle between two 3d vectors
        norm_normal = math.sqrt(normal[0]**2 + normal[1]**2 + normal[2]**2)

        z_axis = [0, 0, 1]
        norm_z_axis = math.sqrt(z_axis[0]**2 + z_axis[1]**2 + z_axis[2]**2)

        theta = math.acos(np.dot(normal, z_axis) / (norm_normal * norm_z_axis))
        # print(normal)
        # print('theta = ' + str(theta))

        # Keep points where angle to z_axis is small enough
        if abs(theta) < 0.36:  # we have a point that belongs to an horizontal surface
            point_cloud_horizontal.points.append(point)
            point_cloud_horizontal.points.append(normal)
            point_cloud_horizontal.points.append(color)

    point_cloud_horizontal.paint_uniform_color([0, 1, 0])

    # --------------------------------------
    # Sequential Plane segmentation
    # --------------------------------------
    point_cloud_remaining = deepcopy(point_cloud_horizontal)
    planes = []

    max_num_planes = 10
    colormap = cm.Pastel1(range(0, max_num_planes))
    idx = 0
    while True:

        num_remaining_points = len(point_cloud_remaining.points)
        print('Searching for plane in point cloud with ' + str(num_remaining_points) + ' points ')
        ps = PlaneSegmentation(point_cloud_remaining)
        ps.findPlane()
        color = colormap[idx, 0:3]
        ps.inliers.paint_uniform_color(color)
        planes.append(ps)

        point_cloud_remaining = ps.outliers

        idx += 1

        num_remaining_points = len(point_cloud_remaining.points)
        print('Found plane ' + str(idx) + ' Remaining ' + str(num_remaining_points) + ' points ')
        if num_remaining_points < 100 or idx >= max_num_planes:
            break

    # --------------------------------------
    # Visualization
    # --------------------------------------

    entities = [point_cloud_original]
    entities.extend([x.inliers for x in planes])
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
