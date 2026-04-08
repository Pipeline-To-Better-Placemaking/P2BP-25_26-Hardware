#!/usr/bin/env python3
"""
Point Cloud Visualization with Surface Reconstruction

This script loads track data and creates a smoothed 3D mesh from point cloud data.
It uses Open3D for surface reconstruction to fill gaps between points.

Install dependencies:
    pip install open3d numpy
"""

import os
import json
import numpy as np
import argparse
from pathlib import Path

try:
    import open3d as o3d
except ImportError:
    print("Error: open3d not installed. Install with: pip install open3d")
    exit(1)


def load_track_data(track_dir='tracks'):
    """Load all track JSON files and extract 3D positions."""
    points = []
    colors = []
    track_ids = []
    
    track_path = Path(track_dir)
    if not track_path.exists():
        print(f"Warning: Track directory '{track_dir}' not found.")
        return None
    
    json_files = list(track_path.glob('*.json'))
    if not json_files:
        print(f"No JSON files found in '{track_dir}'")
        return None
    
    print(f"Loading {len(json_files)} track files...")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            global_id = data.get('global_id', -1)
            positions = data.get('positions', [])
            
            # Extract ground plane coordinates (gx, gy) and use timestamp as Z
            for pos in positions:
                if len(pos) >= 6:
                    frame_idx, timestamp, cx, cy, gx, gy = pos[:6]
                    if gx is not None and gy is not None:
                        # Use ground coordinates as X, Y and normalized timestamp as Z
                        # Or you could use frame_idx as Z for a simpler approach
                        points.append([float(gx), float(gy), float(timestamp)])
                        
                        # Color by track ID
                        color = generate_color(global_id)
                        colors.append(color)
                        track_ids.append(global_id)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue
    
    if not points:
        print("No valid points found in track data.")
        return None
    
    return np.array(points), np.array(colors), track_ids


def load_csv_data(csv_file='positions.csv'):
    """Load positions from CSV file."""
    import csv
    
    points = []
    colors = []
    track_ids = []
    
    if not os.path.exists(csv_file):
        print(f"Warning: CSV file '{csv_file}' not found.")
        return None
    
    print(f"Loading data from {csv_file}...")
    
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                gx = row.get('x_ground', '').strip()
                gy = row.get('y_ground', '').strip()
                gid = int(row.get('global_id', -1))
                timestamp = row.get('timestamp_iso', '')
                
                if gx and gy:
                    points.append([float(gx), float(gy), 0.0])  # Z=0 for ground plane
                    color = generate_color(gid)
                    colors.append(color)
                    track_ids.append(gid)
            except (ValueError, KeyError) as e:
                continue
    
    if not points:
        print("No valid points found in CSV data.")
        return None
    
    return np.array(points), np.array(colors), track_ids


def generate_color(track_id):
    """Generate a consistent color for a track ID."""
    np.random.seed(track_id if track_id >= 0 else 0)
    return np.random.rand(3)


def normalize_points(points):
    """Normalize point coordinates to reasonable scale."""
    if len(points) == 0:
        return points
    
    # Normalize Z axis to match X, Y scale
    points = points.copy()
    x_range = points[:, 0].max() - points[:, 0].min()
    y_range = points[:, 1].max() - points[:, 1].min()
    z_range = points[:, 2].max() - points[:, 2].min()
    
    if z_range > 0:
        # Scale Z to be proportional to X, Y
        avg_xy_range = (x_range + y_range) / 2.0
        if avg_xy_range > 0:
            scale = avg_xy_range / z_range
            points[:, 2] = (points[:, 2] - points[:, 2].min()) * scale
    
    return points


def create_point_cloud(points, colors):
    """Create Open3D point cloud object."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def estimate_normals(pcd, radius=None):
    """Estimate normals for point cloud."""
    if radius is None:
        # Auto-estimate radius based on point cloud density
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = avg_dist * 3.0
    
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=30)
    )
    return pcd


def poisson_reconstruction(pcd, depth=9, width=0, scale=1.1, linear_fit=False):
    """Perform Poisson surface reconstruction."""
    print("Performing Poisson surface reconstruction...")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=depth, width=width, scale=scale, linear_fit=linear_fit
    )
    
    # Remove low density vertices (outliers)
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return mesh


def alpha_shape_reconstruction(pcd, alpha=0.03):
    """Perform Alpha shape surface reconstruction."""
    print("Performing Alpha shape surface reconstruction...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
        pcd, alpha=alpha
    )
    return mesh


def ball_pivoting_reconstruction(pcd, radii=None):
    """Perform Ball Pivoting surface reconstruction."""
    if radii is None:
        # Auto-estimate radii based on point cloud density
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radii = [avg_dist, avg_dist * 2, avg_dist * 4]
    
    print(f"Performing Ball Pivoting reconstruction with radii: {radii}...")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )
    return mesh


def smooth_mesh(mesh, iterations=1):
    """Smooth the mesh using Laplacian smoothing."""
    print("Smoothing mesh...")
    mesh = mesh.filter_smooth_laplacian(number_of_iterations=iterations)
    mesh.compute_vertex_normals()
    return mesh


def visualize_pointcloud_and_mesh(pcd, mesh=None, window_name="Point Cloud"):
    """Visualize point cloud and optionally mesh."""
    vis_list = [pcd]
    
    if mesh is not None:
        # Color mesh based on vertex normals for better visualization
        mesh.compute_vertex_normals()
        vis_list.append(mesh)
    
    o3d.visualization.draw_geometries(
        vis_list,
        window_name=window_name,
        width=1920,
        height=1080,
        point_show_normal=False
    )


def main():
    parser = argparse.ArgumentParser(description='Visualize point cloud with surface reconstruction')
    parser.add_argument('--tracks', type=str, default='tracks', help='Directory containing track JSON files')
    parser.add_argument('--csv', type=str, default='positions.csv', help='CSV file with positions')
    parser.add_argument('--method', type=str, choices=['poisson', 'alpha', 'ball', 'all'], 
                       default='poisson', help='Surface reconstruction method')
    parser.add_argument('--smooth', type=int, default=1, help='Number of smoothing iterations')
    parser.add_argument('--depth', type=int, default=9, help='Poisson reconstruction depth')
    parser.add_argument('--alpha', type=float, default=0.03, help='Alpha shape parameter')
    parser.add_argument('--no-normalize', action='store_true', help='Skip Z-axis normalization')
    
    args = parser.parse_args()
    
    # Load data
    points = None
    colors = None
    track_ids = None
    
    # Try loading from tracks first, then CSV
    result = load_track_data(args.tracks)
    if result is None:
        result = load_csv_data(args.csv)
    
    if result is None:
        print("Error: Could not load any data. Please check your data files.")
        return
    
    points, colors, track_ids = result
    
    print(f"Loaded {len(points)} points")
    print(f"Point cloud bounds: X=[{points[:, 0].min():.2f}, {points[:, 0].max():.2f}], "
          f"Y=[{points[:, 1].min():.2f}, {points[:, 1].max():.2f}], "
          f"Z=[{points[:, 2].min():.2f}, {points[:, 2].max():.2f}]")
    
    # Normalize points if requested
    if not args.no_normalize:
        points = normalize_points(points)
        print("Normalized point coordinates")
    
    # Create point cloud
    pcd = create_point_cloud(points, colors)
    
    # Estimate normals (required for most reconstruction methods)
    print("Estimating normals...")
    pcd = estimate_normals(pcd)
    
    # Visualize original point cloud
    print("\nDisplaying original point cloud. Close window to continue...")
    visualize_pointcloud_and_mesh(pcd, window_name="Original Point Cloud")
    
    # Perform surface reconstruction
    meshes = {}
    
    if args.method in ['poisson', 'all']:
        try:
            mesh_poisson = poisson_reconstruction(pcd, depth=args.depth)
            if args.smooth > 0:
                mesh_poisson = smooth_mesh(mesh_poisson, iterations=args.smooth)
            meshes['Poisson'] = mesh_poisson
        except Exception as e:
            print(f"Poisson reconstruction failed: {e}")
    
    if args.method in ['alpha', 'all']:
        try:
            mesh_alpha = alpha_shape_reconstruction(pcd, alpha=args.alpha)
            if args.smooth > 0:
                mesh_alpha = smooth_mesh(mesh_alpha, iterations=args.smooth)
            meshes['Alpha Shape'] = mesh_alpha
        except Exception as e:
            print(f"Alpha shape reconstruction failed: {e}")
    
    if args.method in ['ball', 'all']:
        try:
            mesh_ball = ball_pivoting_reconstruction(pcd)
            if args.smooth > 0:
                mesh_ball = smooth_mesh(mesh_ball, iterations=args.smooth)
            meshes['Ball Pivoting'] = mesh_ball
        except Exception as e:
            print(f"Ball pivoting reconstruction failed: {e}")
    
    # Visualize results
    if meshes:
        for method_name, mesh in meshes.items():
            print(f"\nDisplaying {method_name} reconstruction. Close window to continue...")
            visualize_pointcloud_and_mesh(pcd, mesh, window_name=f"{method_name} Reconstruction")
            
            # Optionally save mesh
            output_file = f"mesh_{method_name.lower().replace(' ', '_')}.ply"
            try:
                o3d.io.write_triangle_mesh(output_file, mesh)
                print(f"Saved mesh to {output_file}")
            except Exception as e:
                print(f"Could not save mesh: {e}")
    else:
        print("No meshes were successfully created.")


if __name__ == '__main__':
    main()
