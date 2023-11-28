#!/usr/bin/env python3
# Sistemas Avançados de Visão Industrial (SAVI 22-23)
# Miguel Riem Oliveira, DEM, UA


import open3d as o3d


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
    filename = '../Factory/factory.ply'
    print('Loading file ' + filename)
    point_cloud = o3d.io.read_point_cloud(filename)
    print(point_cloud)

    # --------------------------------------
    # Plane segmentation
    # --------------------------------------
    plane_model, inlier_idxs = point_cloud.segment_plane(distance_threshold=0.35,
                                                         ransac_n=3, num_iterations=100)

    a, b, c, d = plane_model
    point_cloud_inliers = point_cloud.select_by_index(inlier_idxs, invert=False)
    point_cloud_inliers.paint_uniform_color([1, 0, 0])

    point_cloud_outliers = point_cloud.select_by_index(inlier_idxs, invert=True)

    o3d.io.write_point_cloud('factory_no_floor.pcd', point_cloud_outliers)
    # o3d.io.write_point_cloud('factory_no_floor.ply', point_cloud_outliers)

    # --------------------------------------
    # Visualization
    # --------------------------------------
    entities = [point_cloud_inliers, point_cloud_outliers]
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
