#!/usr/bin/env python3

import csv
import os
import pickle
import random
import glob
from copy import deepcopy
from random import randint
from turtle import color

import open3d as o3d
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from more_itertools import locate

view = {
	"class_name" : "ViewTrajectory",
	"interval" : 29,
	"is_loop" : False,
	"trajectory" : 
	[
		{
			"boundingbox_max" : [ 6.5291471481323242, 34.024543762207031, 11.225864410400391 ],
			"boundingbox_min" : [ -39.714397430419922, -16.512752532958984, -1.9472264051437378 ],
			"field_of_view" : 60.0,
			"front" : [ 0.54907281448319933, -0.72074094308345071, 0.42314481842352314 ],
			"lookat" : [ -7.4165150225483982, -4.3692552972898397, 4.2418377265036487 ],
			"up" : [ -0.27778678941340029, 0.3201300269334113, 0.90573244696378663 ],
			"zoom" : 0.26119999999999988
		}
	],
	"version_major" : 1,
	"version_minor" : 0
}



class PlaneDetection():
    def __init__(self, point_cloud):

        self.point_cloud = point_cloud

    def colorizeInliers(self, r,g,b):
        self.inlier_cloud.paint_uniform_color([r,g,b]) # paints the plane in red

    def segment(self, distance_threshold=0.04, ransac_n=3, num_iterations=50):

        print('Starting plane detection')
        plane_model, inlier_idxs = self.point_cloud.segment_plane(distance_threshold=distance_threshold, 
                                                    ransac_n=ransac_n,
                                                    num_iterations=num_iterations)
        [self.a, self.b, self.c, self.d] = plane_model

        self.inlier_cloud = self.point_cloud.select_by_index(inlier_idxs)

        outlier_cloud = self.point_cloud.select_by_index(inlier_idxs, invert=True)

        return outlier_cloud

    def __str__(self):
        text = 'Segmented plane from pc with ' + str(len(self.point_cloud.points)) + ' with ' + str(len(self.inlier_cloud.points)) + ' inliers. '
        text += '\nPlane: ' + str(self.a) +  ' x + ' + str(self.b) + ' y + ' + str(self.c) + ' z + ' + str(self.d) + ' = 0' 
        return text


def main():

    # ------------------------------------------
    # Initialization
    # ------------------------------------------
    print("Load a ply point cloud, print it, and render it")

    dataset_path = '/home/mike/Desktop/SAVI_RGBD_Data/RGBD_Scenes/rgbd-scenes-v2/pc'

    # point_cloud_filenames = ['pcds/03.ply', 'pcds/07.ply', 'pcds/10.ply']
    point_cloud_filenames = glob.glob(dataset_path + '/*.ply')
    point_cloud_filename = random.choice(point_cloud_filenames)

    point_cloud_filename = dataset_path + '/11.ply'
    os.system('pcl_ply2pcd ' + point_cloud_filename + ' pcd_point_cloud.pcd')
    point_cloud_original = o3d.io.read_point_cloud('pcd_point_cloud.pcd')
    print('loaded a point cloud with ' + str(len(point_cloud_original.points)))

    point_cloud_downsampled = point_cloud_original.voxel_down_sample(voxel_size=0.01) 
    print('After downsampling point cloud has ' + str(len(point_cloud_downsampled.points)) + ' points')


    number_of_planes = 2
    minimum_number_points = 25
    colormap = cm.Pastel1(list(range(0,number_of_planes)))

    # ------------------------------------------
    # Execution
    # ------------------------------------------

    point_cloud = deepcopy(point_cloud_downsampled) 
    planes = []
    while True: # run consecutive plane detections

        plane = PlaneDetection(point_cloud) # create a new plane instance
        point_cloud = plane.segment() # new point cloud are the outliers of this plane detection
        print(plane)

        # colorization using a colormap
        idx_color = len(planes)
        color = colormap[idx_color, 0:3]
        plane.colorizeInliers(r=color[0], g=color[1], b=color[2])

        planes.append(plane)

        if len(planes) >= number_of_planes: # stop detection planes
            print('Detected planes >= ' + str(number_of_planes))
            break
        elif len(point_cloud.points) < minimum_number_points:
            print('Number of remaining points < ' + str(minimum_number_points))
            break
            

    # Table plane detector
    table_plane = None
    table_plane_mean_y = 1000
    for plane_idx, plane in enumerate(planes):
        center = plane.inlier_cloud.get_center()
        print('Cloud ' + str(plane_idx) + ' has center ' + str(center))
        mean_y = center[1]
        if mean_y < table_plane_mean_y:
            table_plane = plane
            table_plane_mean_y = mean_y

    table_plane.colorizeInliers(r=1, g=0, b=0)


    # Cluster extraction
    cluster_idxs = list(table_plane.inlier_cloud.cluster_dbscan(eps=0.15, min_points=25, print_progress=True))

    # print(cluster_idxs)
    # print(type(cluster_idxs))

    possible_values = list(set(cluster_idxs))
    if -1 in possible_values:
        possible_values.remove(-1)
    print(possible_values)

    largest_cluster_num_points = 0
    largest_cluster_idx = None
    for value in possible_values:
        num_points = cluster_idxs.count(value)
        if num_points > largest_cluster_num_points:
            largest_cluster_idx = value
            largest_cluster_num_points = num_points

    largest_idxs = list(locate(cluster_idxs, lambda x: x == largest_cluster_idx))
    table_cloud = table_plane.inlier_cloud.select_by_index(largest_idxs)
    table_cloud.paint_uniform_color([0,1,0]) # paints the table green

    # ------------------------------------------
    # Visualization
    # ------------------------------------------

    # Create a list of entities to draw
    # entities = [x.inlier_cloud for x in planes]
    entities = [point_cloud]
    entities.append(table_plane.inlier_cloud)
    entities.append(table_cloud)

    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(size=3.0, origin=np.array([0., 0., 0.]))
    entities.append(frame)

    o3d.visualization.draw_geometries(entities,
                                    zoom=view['trajectory'][0]['zoom'],
                                    front=view['trajectory'][0]['front'],
                                    lookat=view['trajectory'][0]['lookat'],
                                    up=view['trajectory'][0]['up'])

if __name__ == "__main__":
    main()
