"""
OmniVGGT Inference Script

This script performs 3D reconstruction from images using the OmniVGGT model.
It supports:
- Loading images and optional depth maps and camera parameters
- Running inference to predict depth and camera poses
- Visualizing results in 3D using viser
- Exporting results to GLB format
"""

import os
import argparse
import threading
import time
from typing import List, Optional

import numpy as np
import torch
import viser
import viser.transforms as viser_tf
from safetensors.torch import load_file
from tqdm import tqdm

from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from omnivggt.utils.misc import select_first_batch
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import (
    apply_sky_segmentation,
    get_world_points_from_depth,
    load_images_and_cameras,
    predictions_to_glb,
)

def viser_wrapper(
    pred_dict: dict,
    port: int = 8080,
    init_conf_threshold: float = 20.0,  # represents percentage (e.g., 20 means filter lowest 20%)
    use_point_map: bool = False,
    background_mode: bool = False,
    mask_sky: bool = False,
    image_folder: Optional[str] = None,
):
    """
    Visualize predicted 3D points and camera poses with viser.

    Args:
        pred_dict (dict):
            {
                "images": (S, 3, H, W)   - Input images,
                "world_points": (S, H, W, 3),
                "world_points_conf": (S, H, W),
                "depth": (S, H, W, 1),
                "depth_conf": (S, H, W),
                "extrinsic": (S, 3, 4),
                "intrinsic": (S, 3, 3),
            }
        port (int): Port number for the viser server.
        init_conf_threshold (float): Initial percentage of low-confidence points to filter out.
        use_point_map (bool): Whether to visualize world_points or use depth-based points.
        background_mode (bool): Whether to run the server in background thread.
        mask_sky (bool): Whether to apply sky segmentation to filter out sky points.
        image_folder (str): Path to the folder containing input images.
    """
    print(f"Starting viser server on port {port}")

    server = viser.ViserServer(host="0.0.0.0", port=port)
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible")

    # Unpack prediction dict
    images = pred_dict["images"]  # (S, 3, H, W)
    depth_map = pred_dict["depth"]  # (S, H, W, 1)
    depth_conf = pred_dict["depth_conf"]  # (S, H, W)
    extrinsics_cam = pred_dict["extrinsic"]  # (S, 3, 4)
    intrinsics_cam = pred_dict["intrinsic"]  # (S, 3, 3)

    # Compute world points from depth if not using the precomputed point map
    if use_point_map and "world_points" in pred_dict:
        # Use precomputed world points if available
        world_points = pred_dict["world_points"]  # (S, H, W, 3)
        conf = pred_dict.get("world_points_conf", depth_conf)  # (S, H, W)
    else:
        # Compute world points by unprojecting depth map
        world_points = unproject_depth_map_to_point_map(depth_map, extrinsics_cam, intrinsics_cam)
        conf = depth_conf

    # Apply sky segmentation if enabled
    if mask_sky and image_folder is not None:
        conf = apply_sky_segmentation(conf, image_folder)

    # Convert images from (S, 3, H, W) to (S, H, W, 3)
    # Then flatten everything for the point cloud
    colors = images.transpose(0, 2, 3, 1)  # now (S, H, W, 3)
    S, H, W, _ = world_points.shape

    # Flatten
    points = world_points.reshape(-1, 3)
    colors_flat = (colors.reshape(-1, 3) * 255).astype(np.uint8)
    conf_flat = conf.reshape(-1)

    cam_to_world_mat = closed_form_inverse_se3(extrinsics_cam)  # shape (S, 4, 4) typically
    # For convenience, we store only (3,4) portion
    cam_to_world = cam_to_world_mat[:, :3, :]

    # Compute scene center and recenter
    scene_center = np.mean(points, axis=0)
    points_centered = points - scene_center
    cam_to_world[..., -1] -= scene_center

    # Store frame indices so we can filter by frame
    frame_indices = np.repeat(np.arange(S), H * W)

    # Build the viser GUI
    gui_show_frames = server.gui.add_checkbox("Show Cameras", initial_value=True)

    # Now the slider represents percentage of points to filter out
    gui_points_conf = server.gui.add_slider(
        "Confidence Percent", min=0, max=100, step=0.1, initial_value=init_conf_threshold
    )

    gui_frame_selector = server.gui.add_dropdown(
        "Show Points from Frames", options=["All"] + [str(i) for i in range(S)], initial_value="All"
    )

    # Create the main point cloud handle
    # Compute the threshold value as the given percentile
    init_threshold_val = np.percentile(conf_flat, init_conf_threshold)
    init_conf_mask = (conf_flat >= init_threshold_val) & (conf_flat > 0.1)
    point_cloud = server.scene.add_point_cloud(
        name="viser_pcd",
        points=points_centered[init_conf_mask],
        colors=colors_flat[init_conf_mask],
        point_size=0.001,
        point_shape="circle",
    )

    # We will store references to frames & frustums so we can toggle visibility
    frames: List[viser.FrameHandle] = []
    frustums: List[viser.CameraFrustumHandle] = []

    def visualize_frames(extrinsics: np.ndarray, images_: np.ndarray) -> None:
        """
        Add camera frames and frustums to the scene.
        extrinsics: (S, 3, 4)
        images_:    (S, 3, H, W)
        """
        # Clear any existing frames or frustums
        for f in frames:
            f.remove()
        frames.clear()
        for fr in frustums:
            fr.remove()
        frustums.clear()

        # Optionally attach a callback that sets the viewpoint to the chosen camera
        def attach_callback(frustum: viser.CameraFrustumHandle, frame: viser.FrameHandle) -> None:
            @frustum.on_click
            def _(_) -> None:
                for client in server.get_clients().values():
                    client.camera.wxyz = frame.wxyz
                    client.camera.position = frame.position

        img_ids = range(S)
        for img_id in tqdm(img_ids):
            cam2world_3x4 = extrinsics[img_id]
            T_world_camera = viser_tf.SE3.from_matrix(cam2world_3x4)

            # Add a small frame axis
            frame_axis = server.scene.add_frame(
                f"frame_{img_id}",
                wxyz=T_world_camera.rotation().wxyz,
                position=T_world_camera.translation(),
                axes_length=0.05,
                axes_radius=0.002,
                origin_radius=0.002,
            )
            frames.append(frame_axis)

            # Convert the image for the frustum
            img = images_[img_id]  # shape (3, H, W)
            img = (img.transpose(1, 2, 0) * 255).astype(np.uint8)
            h, w = img.shape[:2]

            # If you want correct FOV from intrinsics, do something like:
            # fx = intrinsics_cam[img_id, 0, 0]
            # fov = 2 * np.arctan2(h/2, fx)
            # For demonstration, we pick a simple approximate FOV:
            fy = 1.1 * h
            fov = 2 * np.arctan2(h / 2, fy)

            # Add the frustum
            frustum_cam = server.scene.add_camera_frustum(
                f"frame_{img_id}/frustum", fov=fov, aspect=w / h, scale=0.05, image=img, line_width=1.0
            )
            frustums.append(frustum_cam)
            attach_callback(frustum_cam, frame_axis)

    def update_point_cloud() -> None:
        """Update the point cloud based on current GUI selections."""
        # Here we compute the threshold value based on the current percentage
        current_percentage = gui_points_conf.value
        threshold_val = np.percentile(conf_flat, current_percentage)

        print(f"Threshold absolute value: {threshold_val}, percentage: {current_percentage}%")

        conf_mask = (conf_flat >= threshold_val) & (conf_flat > 1e-5)

        if gui_frame_selector.value == "All":
            frame_mask = np.ones_like(conf_mask, dtype=bool)
        else:
            selected_idx = int(gui_frame_selector.value)
            frame_mask = frame_indices == selected_idx

        combined_mask = conf_mask & frame_mask
        point_cloud.points = points_centered[combined_mask]
        point_cloud.colors = colors_flat[combined_mask]

    @gui_points_conf.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_frame_selector.on_update
    def _(_) -> None:
        update_point_cloud()

    @gui_show_frames.on_update
    def _(_) -> None:
        """Toggle visibility of camera frames and frustums."""
        for f in frames:
            f.visible = gui_show_frames.value
        for fr in frustums:
            fr.visible = gui_show_frames.value

    # Add the camera frames to the scene
    visualize_frames(cam_to_world, images)

    print("Starting viser server...")
    # If background_mode is True, spawn a daemon thread so the main thread can continue.
    if background_mode:

        def server_loop():
            while True:
                time.sleep(0.001)

        thread = threading.Thread(target=server_loop, daemon=True)
        thread.start()
    else:
        while True:
            time.sleep(0.01)

    return server

# Command-line argument parser
parser = argparse.ArgumentParser(
    description="OmniVGGT demo with viser for 3D visualization"
)

# Input data arguments
parser.add_argument("--image_folder",type=str,help="Path to folder containing images")
parser.add_argument("--depth_folder",type=str,default=None,help="Path to folder containing depth maps (.npy)")
parser.add_argument("--camera_folder",type=str,default=None,help="Path to folder containing camera files (.txt)")

# Processing options
parser.add_argument("--use_point_map",action="store_true",help="Use point map instead of depth-based points")
parser.add_argument("--mask_sky",action="store_true",help="Apply sky segmentation to filter out sky points")
parser.add_argument("--target_size",type=int,default=518,help="Target size for the images")

# Visualization options
parser.add_argument("--background_mode",action="store_true",help="Run the viser server in background mode")
parser.add_argument("--port",type=int,default=8080,help="Port number for the viser server")
parser.add_argument("--conf_threshold",type=float,default=25.0,help="Initial percentage of low-confidence points to filter out")

# Export options
parser.add_argument("--save_glb",action="store_true",help="Save the output as a GLB file")

def main():
    """
    Main function for the OmniVGGT demo with viser for 3D visualization.

    This function:
    1. Loads the OmniVGGT model
    2. Processes input images from the specified folder
    3. Runs inference to generate 3D points and camera poses
    4. Optionally applies sky segmentation to filter out sky points
    5. Visualizes the results using viser
    """
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Initialize and load OmniVGGT model
    print("Initializing and loading OmniVGGT model...")
    model = OmniVGGT().to(device)
    state_dict = load_file("checkpoints/from132_136k.safetensors")
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # Load input data
    print(f"Loading images from {args.image_folder}...")
    if args.depth_folder is not None:
        print(f"Loading cameras from {args.camera_folder}...")
    if args.camera_folder is not None:
        print(f"Loading depths from {args.depth_folder}...")

    images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices = \
        load_images_and_cameras(
            args.image_folder,
            args.camera_folder,
            args.depth_folder,
            args.target_size
        )

    # Prepare model inputs
    inputs = {
        'images': images.to(device),
        'extrinsics': extrinsics.to(device),
        'intrinsics': intrinsics.to(device),
        'depth': depthmaps.to(device),
        'mask': masks.to(device),
        'depth_gt_index': depth_indices,
        'camera_gt_index': camera_indices
    }

    # Run inference
    print("Running inference...")
    with torch.no_grad():
        predictions = model(**inputs)

    # Convert pose encoding to camera matrices
    print("Converting pose encoding to extrinsic and intrinsic matrices...")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        images.shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Export to GLB if requested
    if args.save_glb:
        print("Exporting scene to GLB...")
        predictions_0 = select_first_batch(predictions)
        get_world_points_from_depth(predictions_0)
        glbscene = predictions_to_glb(
            predictions_0,
            conf_thres=0.0,
            filter_by_frames='All',
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            target_dir=args.image_folder,
            prediction_mode="Predicted Depth",
        )
        glb_path = os.path.join(args.image_folder, 'scene.glb')
        glbscene.export(file_obj=glb_path)
        print(f"Saved GLB file to {glb_path}")

    # Convert predictions to numpy for visualization
    print("Processing model outputs...")
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    # Print visualization mode
    if args.use_point_map:
        print("Visualizing 3D points from point map")
    else:
        print("Visualizing 3D points by unprojecting depth map by cameras")

    if args.mask_sky:
        print("Sky segmentation enabled - will filter out sky points")

    # Start visualization server
    print("Starting viser visualization...")
    viser_server = viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    print("Visualization complete")


if __name__ == "__main__":
    main()