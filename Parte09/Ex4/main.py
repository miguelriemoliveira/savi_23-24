#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA


from copy import deepcopy
from matplotlib import cm
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
    filename = '../Ex2/factory_no_floor.pcd'
    print('Loading file ' + filename)
    point_cloud_original = o3d.io.read_point_cloud(filename)

    # --------------------------------------
    # Downsampling
    # --------------------------------------
    point_cloud_downsampled = point_cloud_original.voxel_down_sample(voxel_size=0.2)
    print(point_cloud_downsampled)

    # --------------------------------------
    # Clustering
    # --------------------------------------
    labels = point_cloud_downsampled.cluster_dbscan(eps=2.0, min_points=50, print_progress=True)

    group_idxs = list(set(labels))
    group_idxs.remove(-1)  # remove last group because its the group on the unassigned points
    num_groups = len(group_idxs)
    colormap = cm.Pastel1(range(0, num_groups))

    group_point_clouds = []
    for group_idx in group_idxs:  # Cycle all groups, i.e.,

        group_points_idxs = list(locate(labels, lambda x: x == group_idx))

        group_point_cloud = point_cloud_downsampled.select_by_index(group_points_idxs, invert=False)

        color = colormap[group_idx, 0:3]
        # group_point_cloud.paint_uniform_color(color)
        group_point_clouds.append(group_point_cloud)

    # --------------------------------------
    # Find the largest group
    # --------------------------------------
    max_group_point_cloud = None
    for group_point_cloud in group_point_clouds:

        if max_group_point_cloud is None:  # if there is no largest, keep this one
            max_group_point_cloud = group_point_cloud
        elif len(group_point_cloud.points) > len(max_group_point_cloud.points):
            max_group_point_cloud = group_point_cloud

    o3d.io.write_point_cloud('factory_isolated.pcd', max_group_point_cloud)

    # Paint largest group in bright red
    max_group_point_cloud.paint_uniform_color((1, 0, 0))

    # --------------------------------------
    # Visualization
    # --------------------------------------
    entities = group_point_clouds
    o3d.visualization.draw_geometries(entities,
                                      zoom=0.3412,
                                      front=view['trajectory'][0]['front'],
                                      lookat=view['trajectory'][0]['lookat'],
                                      up=view['trajectory'][0]['up'])

    # --------------------------------------
    # Termination
    # --------------------------------------


if __name__ == "__main__":
    main()
