#!/usr/bin/env python3
"""
Fast3R to COLMAP Converter

Converts Fast3R 3D reconstruction outputs to COLMAP sparse format.
Creates cameras.txt, images.txt, and points3D.txt files.
"""

import numpy as np
import os
from pathlib import Path


def convert_pose_c2w_to_w2c(c2w_matrix):
    """Convert camera-to-world pose to world-to-camera pose."""
    return np.linalg.inv(c2w_matrix)


def rotation_matrix_to_quaternion(R):
    """Convert rotation matrix to quaternion (w, x, y, z)."""
    trace = np.trace(R)
    
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2  # s = 4 * qw
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2  # s = 4 * qx
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2  # s = 4 * qy
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2  # s = 4 * qz
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s
    
    return np.array([qw, qx, qy, qz])


def sparsify_point_cloud(point_cloud, step=8):
    """Sample points from dense point cloud (every step-th pixel)."""
    H, W, _ = point_cloud.shape
    
    # Sample every step-th pixel
    y_coords, x_coords = np.meshgrid(
        np.arange(0, H, step),
        np.arange(0, W, step),
        indexing='ij'
    )
    
    # Get sampled points and their 2D coordinates
    sampled_points = point_cloud[y_coords, x_coords]
    
    # Flatten to get list of points and coordinates
    points_3d = sampled_points.reshape(-1, 3)
    points_2d = np.column_stack([x_coords.flatten(), y_coords.flatten()])
    
    # Filter out invalid points (e.g., zeros or NaN)
    valid_mask = ~np.any(np.isnan(points_3d), axis=1) & ~np.all(points_3d == 0, axis=1)
    
    return points_3d[valid_mask], points_2d[valid_mask]


class Fast3RToColmapConverter:
    """Converter from Fast3R format to COLMAP sparse reconstruction."""
    
    def __init__(self, output_dir):
        """Initialize converter with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data structures for COLMAP format
        self.cameras = {}
        self.images = {}
        self.points3d = {}
        self.point_counter = 1
        
    def convert(self, poses, point_clouds, image_paths, focal_lengths):
        """
        Main conversion function.
        
        Args:
            poses: List of 4x4 numpy arrays (camera-to-world)
            point_clouds: List of dense arrays with shape (H, W, 3)
            image_paths: List of strings (image file paths)
            focal_lengths: List of float values per camera
        """
        print(f"Converting {len(poses)} cameras to COLMAP format...")
        
        # Create cameras
        self._create_cameras(focal_lengths, point_clouds)
        
        # Convert poses and create image entries
        self._convert_poses(poses, image_paths)
        
        # Process point clouds
        self._process_points(point_clouds, poses)
        
        # Write output files
        self._write_files()
        
        print(f"Conversion complete. Files written to {self.output_dir}")
        
    def _create_cameras(self, focal_lengths, point_clouds):
        """Create COLMAP camera models (SIMPLE_RADIAL)."""
        for i, (focal_length, point_cloud) in enumerate(zip(focal_lengths, point_clouds)):
            camera_id = i + 1
            H, W, _ = point_cloud.shape
            
            # SIMPLE_RADIAL model: f, cx, cy, k1
            cx = W / 2.0
            cy = H / 2.0
            k1 = 0.0  # No distortion
            
            self.cameras[camera_id] = {
                'model': 'SIMPLE_RADIAL',
                'width': W,
                'height': H,
                'params': [focal_length, cx, cy, k1]
            }
            
    def _convert_poses(self, poses, image_paths):
        """Convert poses and create image entries."""
        for i, (pose, image_path) in enumerate(zip(poses, image_paths)):
            image_id = i + 1
            camera_id = i + 1
            
            # Convert camera-to-world to world-to-camera
            w2c = convert_pose_c2w_to_w2c(pose)
            
            # Extract rotation and translation
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            
            # Convert rotation matrix to quaternion
            quat = rotation_matrix_to_quaternion(R)
            
            # Get image name
            image_name = os.path.basename(image_path)
            
            self.images[image_id] = {
                'qvec': quat,
                'tvec': t,
                'camera_id': camera_id,
                'name': image_name,
                'points2d': []  # Will be filled during point processing
            }
            
    def _process_points(self, point_clouds, poses):
        """Convert dense clouds to sparse points with observations."""
        print("Processing point clouds...")
        
        all_points_3d = []
        all_observations = []
        
        for i, (point_cloud, pose) in enumerate(zip(point_clouds, poses)):
            image_id = i + 1
            
            # Sparsify point cloud
            points_3d_local, points_2d = sparsify_point_cloud(point_cloud)
            
            if len(points_3d_local) == 0:
                continue
                
            # Transform points to world coordinates
            c2w = pose
            points_3d_world = (c2w[:3, :3] @ points_3d_local.T + c2w[:3, 3:4]).T
            
            # Store points and observations
            start_idx = len(all_points_3d)
            all_points_3d.extend(points_3d_world)
            
            for j, point_2d in enumerate(points_2d):
                point3d_id = start_idx + j + 1
                all_observations.append({
                    'point3d_id': point3d_id,
                    'image_id': image_id,
                    'point2d': point_2d
                })
                
                # Add to image's 2D points
                self.images[image_id]['points2d'].append({
                    'xy': point_2d,
                    'point3d_id': point3d_id
                })
        
        # Create 3D points with observations
        for i, point_3d in enumerate(all_points_3d):
            point3d_id = i + 1
            
            # Find observations for this point
            observations = [obs for obs in all_observations if obs['point3d_id'] == point3d_id]
            
            # Default color (gray)
            color = [128, 128, 128]
            error = 0.0
            
            self.points3d[point3d_id] = {
                'xyz': point_3d,
                'rgb': color,
                'error': error,
                'track': observations
            }
            
        print(f"Created {len(self.points3d)} 3D points")
        
    def _write_files(self):
        """Write cameras.txt, images.txt, points3D.txt."""
        self._write_cameras()
        self._write_images()
        self._write_points3d()
        
    def _write_cameras(self):
        """Write cameras.txt file."""
        with open(self.output_dir / 'cameras.txt', 'w') as f:
            f.write("# Camera list with one line of data per camera:\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            f.write("# Number of cameras: {}\n".format(len(self.cameras)))
            
            for camera_id, camera in self.cameras.items():
                params_str = ' '.join(map(str, camera['params']))
                f.write(f"{camera_id} {camera['model']} {camera['width']} {camera['height']} {params_str}\n")
                
    def _write_images(self):
        """Write images.txt file."""
        with open(self.output_dir / 'images.txt', 'w') as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write("# Number of images: {}, mean observations per image: {}\n".format(
                len(self.images), 
                np.mean([len(img['points2d']) for img in self.images.values()]) if self.images else 0
            ))
            
            for image_id, image in self.images.items():
                # Write image line
                qvec_str = ' '.join(map(str, image['qvec']))
                tvec_str = ' '.join(map(str, image['tvec']))
                f.write(f"{image_id} {qvec_str} {tvec_str} {image['camera_id']} {image['name']}\n")
                
                # Write points2d line
                if image['points2d']:
                    points2d_str = ' '.join([
                        f"{pt['xy'][0]} {pt['xy'][1]} {pt['point3d_id']}"
                        for pt in image['points2d']
                    ])
                    f.write(f"{points2d_str}\n")
                else:
                    f.write("\n")
                    
    def _write_points3d(self):
        """Write points3D.txt file."""
        with open(self.output_dir / 'points3D.txt', 'w') as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points: {}, mean track length: {}\n".format(
                len(self.points3d),
                np.mean([len(pt['track']) for pt in self.points3d.values()]) if self.points3d else 0
            ))
            
            for point3d_id, point in self.points3d.items():
                xyz_str = ' '.join(map(str, point['xyz']))
                rgb_str = ' '.join(map(str, point['rgb']))
                
                # Create track string (IMAGE_ID, POINT2D_IDX pairs)
                track_str = ''
                for obs in point['track']:
                    # Find the index of this point in the image's points2d list
                    image = self.images[obs['image_id']]
                    point2d_idx = -1
                    for idx, pt2d in enumerate(image['points2d']):
                        if pt2d['point3d_id'] == point3d_id:
                            point2d_idx = idx
                            break
                    
                    if point2d_idx >= 0:
                        track_str += f" {obs['image_id']} {point2d_idx}"
                
                f.write(f"{point3d_id} {xyz_str} {rgb_str} {point['error']}{track_str}\n")


# Example usage
if __name__ == "__main__":
    # Example data (replace with actual Fast3R outputs)
    """
    # Example usage:
    converter = Fast3RToColmapConverter("./colmap_output")
    converter.convert(
        poses=fast3r_poses,           # List of 4x4 matrices
        point_clouds=fast3r_clouds,   # List of (H,W,3) arrays  
        image_paths=image_files,      # List of image paths
        focal_lengths=focal_lengths   # List of focal length values
    )
    """
    
    print("Fast3R to COLMAP converter ready.")
    print("Import and use the Fast3RToColmapConverter class with your Fast3R data.")