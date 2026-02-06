
import pycolmap

# Load reconstruction from COLMAP output folder
# reconstruction = pycolmap.Reconstruction("assets/emi_pics/results_superpoint+lightglue_matching_lowres_quality_medium/reconstruction/0")
# reconstruction = pycolmap.Reconstruction("assets/emi_pics/results_loftr_matching_lowres_quality_high_0011/reconstruction/0") # maybe about half the face is there0
# reconstruction = pycolmap.Reconstruction("assets/emi_rock/results_disk+lightglue_matching_lowres_quality_high_0028/reconstruction/0") # very good rock
# reconstruction = pycolmap.Reconstruction("assets/emi_pics/results_disk+lightglue_matching_lowres_quality_high/reconstruction/0") #super bad
# reconstruction = pycolmap.Reconstruction("assets/emi_bear/results_disk+lightglue_matching_lowres_quality_high/reconstruction/0") # only 4 images -- creates a plane
# reconstruction = pycolmap.Reconstruction("assets/emi_bear/results_loftr_matching_lowres_quality_high_003/reconstruction/0") # 3 images better than disk but also not amazing
reconstruction = pycolmap.Reconstruction("assets/emi_jedd/results_superpoint+lightglue_matching_lowres_quality_high_0031/reconstruction/0") # jedd's fae with all 31 images looks good
# reconstruction = pycolmap.Reconstruction("assets/example_nadar/results_loftr_matching_lowres_quality_high_0118/reconstruction/0")
# reconstruction = pycolmap.Reconstruction("assets/emi_sword/results_superpoint+lightglue_matching_lowres_quality_high_0013/reconstruction/1") # this is actually the camera behind it
# reconstruction = pycolmap.Reconstruction("assets/emi_carseat/results_superpoint+lightglue_matching_lowres_quality_high0128/reconstruction/0")
# reconstruction = pycolmap.Reconstruction("assets/emi_carseat/results_loftr_matching_lowres_quality_high_0023/reconstruction/0")
reconstruction = pycolmap.Reconstruction("assets/emi_carseat/results_loftr_matching_lowres_quality_high_0059/reconstruction/0") # maybe the car seat is too complicated
reconstruction = pycolmap.Reconstruction("assets/emi_carseat/results_roma_matching_lowres_quality_high_0066/reconstruction/0")
points3D = reconstruction.points3D  # dictionary: {point_id: Point3D}

import numpy as np

# Extract XYZ and RGB
xyz = np.array([p.xyz for p in points3D.values()])
# rgb = np.array([p.rgb for p in points3D.values()])

import open3d as o3d

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(xyz)
# pcd.colors = o3d.utility.Vector3dVector(rgb)

o3d.visualization.draw_geometries([pcd])

for image in reconstruction.images.values():
    # pose = image.cam_from_world.matrix()

    pose_3x4 = image.cam_from_world.matrix()  # shape (3,4)
    pose_4x4 = np.eye(4)
    pose_4x4[:3, :] = pose_3x4  # copy rotation + translation

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    frame.transform(pose_4x4)


    # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    # frame.transform(pose)
    # Add to visualization list