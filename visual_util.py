# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Visual Utilities for OmniVGGT

This module provides utilities for:
- Converting predictions to 3D scenes (GLB format)
- Loading and processing images and camera parameters
- Sky segmentation for filtering
- 3D scene visualization and manipulation
"""

import copy
import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import matplotlib
import numpy as np
import onnxruntime
import PIL
import requests
import torch
import trimesh
import viser
import viser.transforms as viser_tf
from PIL import Image
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import omnivggt.datasets.utils.cropping as cropping
from omnivggt.utils.geometry import closed_form_inverse_se3, unproject_depth_map_to_point_map
from omnivggt.utils.image import ImgNorm, imread_cv2
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri

def get_world_points_from_depth(predictions, gt_scale=1):
    """
    Generate 3D world points from depth predictions.

    Args:
        predictions (dict): Prediction dictionary containing pose_enc, depth, and images
        gt_scale (float): Scale factor for ground truth depth (default: 1)

    Returns:
        None: Updates predictions dict in-place with world_points_from_depth
    """
    # Convert pose encoding to camera matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"],
        predictions["images"].shape[-2:]
    )
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy arrays
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)

    # Unproject depth map to 3D world points
    depth_map = predictions["depth"] * gt_scale
    world_points = unproject_depth_map_to_point_map(
        depth_map,
        predictions["extrinsic"],
        predictions["intrinsic"]
    )
    predictions["world_points_from_depth"] = world_points

def predictions_to_glb(
    predictions,
    conf_thres=50.0,
    filter_by_frames="all",
    mask_black_bg=False,
    mask_white_bg=False,
    show_cam=True,
    mask_sky=False,
    target_dir=None,
    prediction_mode="Predicted Pointmap",
) -> trimesh.Scene:
    """
    Converts VGGT predictions to a 3D scene represented as a GLB file.

    Args:
        predictions (dict): Dictionary containing model predictions with keys:
            - world_points: 3D point coordinates (S, H, W, 3)
            - world_points_conf: Confidence scores (S, H, W)
            - images: Input images (S, H, W, 3)
            - extrinsic: Camera extrinsic matrices (S, 3, 4)
        conf_thres (float): Percentage of low-confidence points to filter out (default: 50.0)
        filter_by_frames (str): Frame filter specification (default: "all")
        mask_black_bg (bool): Mask out black background pixels (default: False)
        mask_white_bg (bool): Mask out white background pixels (default: False)
        show_cam (bool): Include camera visualization (default: True)
        mask_sky (bool): Apply sky segmentation mask (default: False)
        target_dir (str): Output directory for intermediate files (default: None)
        prediction_mode (str): Prediction mode selector (default: "Predicted Pointmap")

    Returns:
        trimesh.Scene: Processed 3D scene containing point cloud and cameras

    Raises:
        ValueError: If input predictions structure is invalid
    """
    if not isinstance(predictions, dict):
        raise ValueError("predictions must be a dictionary")

    if conf_thres is None:
        conf_thres = 10.0

    selected_frame_idx = None
    if filter_by_frames != "all" and filter_by_frames != "All":
        try:
            # Extract the index part before the colon
            selected_frame_idx = int(filter_by_frames.split(":")[0])
        except (ValueError, IndexError):
            pass

    if "Pointmap" in prediction_mode:
        if "world_points" in predictions:
            pred_world_points = predictions["world_points"]  # No batch dimension to remove
            pred_world_points_conf = predictions.get("world_points_conf", np.ones_like(pred_world_points[..., 0]))
        else:
            pred_world_points = predictions["world_points_from_depth"]
            pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    else:
        pred_world_points = predictions["world_points_from_depth"]
        pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))

    # Get images from predictions
    images = predictions["images"]
    # Use extrinsic matrices instead of pred_extrinsic_list
    camera_matrices = predictions["extrinsic"]

    if mask_sky:
        if target_dir is not None:
            import onnxruntime

            skyseg_session = None
            target_dir_images = target_dir + "/images"
            image_list = sorted(os.listdir(target_dir_images))
            sky_mask_list = []

            # Get the shape of pred_world_points_conf to match
            S, H, W = (
                pred_world_points_conf.shape
                if hasattr(pred_world_points_conf, "shape")
                else (len(images), images.shape[1], images.shape[2])
            )

            # Download skyseg.onnx if it doesn't exist
            if not os.path.exists("skyseg.onnx"):
                print("Downloading skyseg.onnx...")
                download_file_from_url(
                    "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx", "skyseg.onnx"
                )

            for i, image_name in enumerate(image_list):
                image_filepath = os.path.join(target_dir_images, image_name)
                mask_filepath = os.path.join(target_dir, "sky_masks", image_name)

                # Check if mask already exists
                if os.path.exists(mask_filepath):
                    # Load existing mask
                    sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
                else:
                    # Generate new mask
                    if skyseg_session is None:
                        skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
                    sky_mask = segment_sky(image_filepath, skyseg_session, mask_filepath)

                # Resize mask to match H×W if needed
                if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
                    sky_mask = cv2.resize(sky_mask, (W, H))

                sky_mask_list.append(sky_mask)

            # Convert list to numpy array with shape S×H×W
            sky_mask_array = np.array(sky_mask_list)

            # Apply sky mask to confidence scores
            sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
            pred_world_points_conf = pred_world_points_conf * sky_mask_binary

    if selected_frame_idx is not None:
        pred_world_points = pred_world_points[selected_frame_idx][None]
        pred_world_points_conf = pred_world_points_conf[selected_frame_idx][None]
        images = images[selected_frame_idx][None]
        camera_matrices = camera_matrices[selected_frame_idx][None]

    vertices_3d = pred_world_points.reshape(-1, 3)
    # Handle different image formats - check if images need transposing
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:  # Assume already in NHWC format
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    conf = pred_world_points_conf.reshape(-1)
    # Convert percentage threshold to actual confidence value
    if conf_thres == 0.0:
        conf_threshold = 0.0
    else:
        conf_threshold = np.percentile(conf, conf_thres)

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)

    if mask_black_bg:
        black_bg_mask = colors_rgb.sum(axis=1) >= 16
        conf_mask = conf_mask & black_bg_mask

    if mask_white_bg:
        # Filter out white background pixels (RGB values close to white)
        # Consider pixels white if all RGB values are above 240
        white_bg_mask = ~((colors_rgb[:, 0] > 240) & (colors_rgb[:, 1] > 240) & (colors_rgb[:, 2] > 240))
        conf_mask = conf_mask & white_bg_mask

    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    if vertices_3d is None or np.asarray(vertices_3d).size == 0:
        vertices_3d = np.array([[1, 0, 0]])
        colors_rgb = np.array([[255, 255, 255]])
        scene_scale = 1
    else:
        # Calculate the 5th and 95th percentiles along each axis
        lower_percentile = np.percentile(vertices_3d, 5, axis=0)
        upper_percentile = np.percentile(vertices_3d, 95, axis=0)

        # Calculate the diagonal length of the percentile bounding box
        scene_scale = np.linalg.norm(upper_percentile - lower_percentile)

    colormap = matplotlib.colormaps.get_cmap("gist_rainbow")

    # Initialize a 3D scene
    scene_3d = trimesh.Scene()

    # Add point cloud data to the scene
    point_cloud_data = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)

    scene_3d.add_geometry(point_cloud_data)

    # Prepare 4x4 matrices for camera extrinsics
    num_cameras = len(camera_matrices)
    extrinsics_matrices = np.zeros((num_cameras, 4, 4))
    extrinsics_matrices[:, :3, :4] = camera_matrices
    extrinsics_matrices[:, 3, 3] = 1

    if show_cam:
        # Add camera models to the scene
        for i in range(num_cameras):
            world_to_camera = extrinsics_matrices[i]
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba_color = colormap(i / num_cameras)
            current_color = tuple(int(255 * x) for x in rgba_color[:3])

            integrate_camera_into_scene(scene_3d, camera_to_world, current_color, scene_scale)

    # Align scene to the observation of the first camera
    scene_3d = apply_scene_alignment(scene_3d, extrinsics_matrices)

    return scene_3d


def integrate_camera_into_scene(
    scene: trimesh.Scene,
    transform: np.ndarray,
    face_colors: tuple,
    scene_scale: float,
):
    """
    Integrates a fake camera mesh into the 3D scene.

    Args:
        scene (trimesh.Scene): The 3D scene to add the camera model.
        transform (np.ndarray): Transformation matrix for camera positioning.
        face_colors (tuple): Color of the camera face.
        scene_scale (float): Scale of the scene.
    """

    cam_width = scene_scale * 0.05
    cam_height = scene_scale * 0.1

    # Create cone shape for camera
    rot_45_degree = np.eye(4)
    rot_45_degree[:3, :3] = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    rot_45_degree[2, 3] = -cam_height

    opengl_transform = get_opengl_conversion_matrix()
    # Combine transformations
    complete_transform = transform @ opengl_transform @ rot_45_degree
    camera_cone_shape = trimesh.creation.cone(cam_width, cam_height, sections=4)

    # Generate mesh for the camera
    slight_rotation = np.eye(4)
    slight_rotation[:3, :3] = Rotation.from_euler("z", 2, degrees=True).as_matrix()

    vertices_combined = np.concatenate(
        [
            camera_cone_shape.vertices,
            0.95 * camera_cone_shape.vertices,
            transform_points(slight_rotation, camera_cone_shape.vertices),
        ]
    )
    vertices_transformed = transform_points(complete_transform, vertices_combined)

    mesh_faces = compute_camera_faces(camera_cone_shape)

    # Add the camera mesh to the scene
    camera_mesh = trimesh.Trimesh(vertices=vertices_transformed, faces=mesh_faces)
    camera_mesh.visual.face_colors[:, :3] = face_colors
    scene.add_geometry(camera_mesh)


def apply_scene_alignment(scene_3d: trimesh.Scene, extrinsics_matrices: np.ndarray) -> trimesh.Scene:
    """
    Aligns the 3D scene based on the extrinsics of the first camera.

    Args:
        scene_3d (trimesh.Scene): The 3D scene to be aligned.
        extrinsics_matrices (np.ndarray): Camera extrinsic matrices.

    Returns:
        trimesh.Scene: Aligned 3D scene.
    """
    # Set transformations for scene alignment
    opengl_conversion_matrix = get_opengl_conversion_matrix()

    # Rotation matrix for alignment (180 degrees around the y-axis)
    align_rotation = np.eye(4)
    align_rotation[:3, :3] = Rotation.from_euler("y", 180, degrees=True).as_matrix()

    # Apply transformation
    initial_transformation = np.linalg.inv(extrinsics_matrices[0]) @ opengl_conversion_matrix @ align_rotation
    scene_3d.apply_transform(initial_transformation)
    return scene_3d


def get_opengl_conversion_matrix() -> np.ndarray:
    """
    Constructs and returns the OpenGL conversion matrix.

    Returns:
        numpy.ndarray: A 4x4 OpenGL conversion matrix.
    """
    # Create an identity matrix
    matrix = np.identity(4)

    # Flip the y and z axes
    matrix[1, 1] = -1
    matrix[2, 2] = -1

    return matrix


def transform_points(transformation: np.ndarray, points: np.ndarray, dim: int = None) -> np.ndarray:
    """
    Applies a 4x4 transformation to a set of points.

    Args:
        transformation (np.ndarray): Transformation matrix.
        points (np.ndarray): Points to be transformed.
        dim (int, optional): Dimension for reshaping the result.

    Returns:
        np.ndarray: Transformed points.
    """
    points = np.asarray(points)
    initial_shape = points.shape[:-1]
    dim = dim or points.shape[-1]

    # Apply transformation
    transformation = transformation.swapaxes(-1, -2)  # Transpose the transformation matrix
    points = points @ transformation[..., :-1, :] + transformation[..., -1:, :]

    # Reshape the result
    result = points[..., :dim].reshape(*initial_shape, dim)
    return result


def compute_camera_faces(cone_shape: trimesh.Trimesh) -> np.ndarray:
    """
    Computes the faces for the camera mesh.

    Args:
        cone_shape (trimesh.Trimesh): The shape of the camera cone.

    Returns:
        np.ndarray: Array of faces for the camera mesh.
    """
    # Create pseudo cameras
    faces_list = []
    num_vertices_cone = len(cone_shape.vertices)

    for face in cone_shape.faces:
        if 0 in face:
            continue
        v1, v2, v3 = face
        v1_offset, v2_offset, v3_offset = face + num_vertices_cone
        v1_offset_2, v2_offset_2, v3_offset_2 = face + 2 * num_vertices_cone

        faces_list.extend(
            [
                (v1, v2, v2_offset),
                (v1, v1_offset, v3),
                (v3_offset, v2, v3),
                (v1, v2, v2_offset_2),
                (v1, v1_offset_2, v3),
                (v3_offset_2, v2, v3),
            ]
        )

    faces_list += [(v3, v2, v1) for v1, v2, v3 in faces_list]
    return np.array(faces_list)


def segment_sky(image_path, onnx_session, mask_filename=None):
    """
    Segments sky from an image using an ONNX model.
    Thanks for the great model provided by https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing

    Args:
        image_path: Path to input image
        onnx_session: ONNX runtime session with loaded model
        mask_filename: Path to save the output mask

    Returns:
        np.ndarray: Binary mask where 255 indicates non-sky regions
    """

    assert mask_filename is not None
    image = cv2.imread(image_path)

    result_map = run_skyseg(onnx_session, [320, 320], image)
    # resize the result_map to the original image size
    result_map_original = cv2.resize(result_map, (image.shape[1], image.shape[0]))

    # Fix: Invert the mask so that 255 = non-sky, 0 = sky
    # The model outputs low values for sky, high values for non-sky
    output_mask = np.zeros_like(result_map_original)
    output_mask[result_map_original < 32] = 255  # Use threshold of 32

    os.makedirs(os.path.dirname(mask_filename), exist_ok=True)
    cv2.imwrite(mask_filename, output_mask)
    return output_mask


def run_skyseg(onnx_session, input_size, image):
    """
    Runs sky segmentation inference using ONNX model.

    Args:
        onnx_session: ONNX runtime session
        input_size: Target size for model input (width, height)
        image: Input image in BGR format

    Returns:
        np.ndarray: Segmentation mask
    """

    # Pre process:Resize, BGR->RGB, Transpose, PyTorch standardization, float32 cast
    temp_image = copy.deepcopy(image)
    resize_image = cv2.resize(temp_image, dsize=(input_size[0], input_size[1]))
    x = cv2.cvtColor(resize_image, cv2.COLOR_BGR2RGB)
    x = np.array(x, dtype=np.float32)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    x = (x / 255 - mean) / std
    x = x.transpose(2, 0, 1)
    x = x.reshape(-1, 3, input_size[0], input_size[1]).astype("float32")

    # Inference
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    onnx_result = onnx_session.run([output_name], {input_name: x})

    # Post process
    onnx_result = np.array(onnx_result).squeeze()
    min_value = np.min(onnx_result)
    max_value = np.max(onnx_result)
    onnx_result = (onnx_result - min_value) / (max_value - min_value)
    onnx_result *= 255
    onnx_result = onnx_result.astype("uint8")

    return onnx_result


def download_file_from_url(url, filename):
    """Downloads a file from a Hugging Face model repo, handling redirects."""
    try:
        # Get the redirect URL
        response = requests.get(url, allow_redirects=False)
        response.raise_for_status()  # Raise HTTPError for bad requests (4xx or 5xx)

        if response.status_code == 302:  # Expecting a redirect
            redirect_url = response.headers["Location"]
            response = requests.get(redirect_url, stream=True)
            response.raise_for_status()
        else:
            print(f"Unexpected status code: {response.status_code}")
            return

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Downloaded {filename} successfully.")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading file: {e}")


def load_image_data(
    image_path: str,
    depth_path: Optional[str] = None,
    camera_path: Optional[str] = None,
    resolution: Tuple[int, int] = (518, 378)
) -> Dict:
    """
    Load a single image with optional depth and camera information.
    
    Args:
        image_path: Path to RGB image
        depth_path: Optional path to depth map (.npy file)
        camera_path: Optional path to camera info (.npz file with 'pose' and 'intrinsics')
        resolution: Target resolution (W, H)
    
    Returns:
        dict with keys: 'image', 'depth', 'extrinsic', 'intrinsic', 'mask'
        Values can be None if not provided
    """
    # Load RGB image
    rgb_image = imread_cv2(image_path, cv2.IMREAD_COLOR)
    
    # Load depth if provided
    depthmap = None
    if depth_path is not None and os.path.exists(depth_path):
        try:
            depthmap = np.load(depth_path).astype(np.float32)
            depthmap[~np.isfinite(depthmap)] = 0  # handle invalid values
            print(f"  Loaded depth from {depth_path}")
        except Exception as e:
            print(f"  [WARN] Failed to load depth from {depth_path}: {e}")
            depthmap = None
    
    # Load camera info if provided
    extrinsic = None
    intrinsic = None
    if camera_path is not None and os.path.exists(camera_path):
        try:
            camera_info = np.load(camera_path)
            extrinsic = np.array(camera_info['pose'], dtype=np.float32)  # (4,4) cam-to-world
            intrinsic = np.array(camera_info['intrinsics'], dtype=np.float32)  # (3,3)
            print(f"  Loaded camera from {camera_path}")
        except Exception as e:
            print(f"  [WARN] Failed to load camera from {camera_path}: {e}")
            extrinsic = None
            intrinsic = None
    
    # Resize (simple version without complex cropping)
    rgb_image, depthmap, intrinsic = simple_resize_image(
        rgb_image, depthmap, intrinsic, resolution
    )
    
    # Convert extrinsic from cam-to-world to world-to-cam if provided
    if extrinsic is not None:
        extrinsic = closed_form_inverse_se3(extrinsic[None])[0]  # invert SE3
    
    # For now, we don't compute point mask (can be added later if needed)
    point_mask = None
    
    # Normalize image
    image_tensor = ImgNorm(np.array(rgb_image))
    
    return {
        'image': image_tensor,
        'depth': depthmap,
        'extrinsic': extrinsic,
        'intrinsic': intrinsic,
        'mask': point_mask,
        'rgb_raw': rgb_image  # keep for visualization
    }
    
def simple_resize_image(image, depthmap, intrinsics, resolution):
    """
    Simple resize of image and depthmap, adjusting intrinsics if provided.
    
    Args:
        image: PIL Image or numpy array
        depthmap: numpy array or None
        intrinsics: (3,3) camera intrinsics or None
        resolution: (W, H) tuple
    
    Returns:
        image, depthmap, intrinsics (all appropriately resized, or None if input was None)
    """
    if not isinstance(image, PIL.Image.Image):
        image = PIL.Image.fromarray(image)
    
    W_orig, H_orig = image.size
    W_target, H_target = resolution
    
    # Resize image
    image = image.resize(resolution, PIL.Image.Resampling.BICUBIC)
    
    # Resize depthmap if provided
    if depthmap is not None:
        depthmap = cv2.resize(depthmap, resolution, interpolation=cv2.INTER_NEAREST)
    
    # Adjust intrinsics if provided
    if intrinsics is not None:
        scale_x = W_target / W_orig
        scale_y = H_target / H_orig
        intrinsics = intrinsics.copy()
        intrinsics[0, :] *= scale_x  # Scale fx and cx
        intrinsics[1, :] *= scale_y  # Scale fy and cy
    
    return image, depthmap, intrinsics


def apply_sky_segmentation(conf: np.ndarray, image_folder: str) -> np.ndarray:
    """
    Apply sky segmentation to confidence scores.

    Args:
        conf (np.ndarray): Confidence scores with shape (S, H, W)
        image_folder (str): Path to the folder containing input images

    Returns:
        np.ndarray: Updated confidence scores with sky regions masked out
    """
    S, H, W = conf.shape
    sky_masks_dir = image_folder.rstrip("/") + "_sky_masks"
    os.makedirs(sky_masks_dir, exist_ok=True)

    # Download skyseg.onnx if it doesn't exist
    if not os.path.exists("skyseg.onnx"):
        print("Downloading skyseg.onnx...")
        download_file_from_url(
            "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
            "skyseg.onnx"
        )

    skyseg_session = onnxruntime.InferenceSession("skyseg.onnx")
    image_files = sorted(glob.glob(os.path.join(image_folder, "*")))
    sky_mask_list = []

    print("Generating sky masks...")
    for i, image_path in enumerate(tqdm(image_files[:S])):
        image_name = os.path.basename(image_path)
        mask_filepath = os.path.join(sky_masks_dir, image_name)

        # Load existing mask or generate new one
        if os.path.exists(mask_filepath):
            sky_mask = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
        else:
            sky_mask = segment_sky(image_path, skyseg_session, mask_filepath)

        # Resize mask to match confidence map dimensions
        if sky_mask.shape[0] != H or sky_mask.shape[1] != W:
            sky_mask = cv2.resize(sky_mask, (W, H))

        sky_mask_list.append(sky_mask)

    # Convert list to numpy array with shape (S, H, W)
    sky_mask_array = np.array(sky_mask_list)

    # Apply sky mask to confidence scores (0 = sky, 1 = non-sky)
    sky_mask_binary = (sky_mask_array > 0.1).astype(np.float32)
    conf = conf * sky_mask_binary

    print("Sky segmentation applied successfully")
    return conf

def load_images_and_cameras(
    image_folder: str,
    camera_folder: Optional[str] = None,
    depth_folder: Optional[str] = None,
    target_size: int = 518,
    max_depth: float = 100,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
    """
    Load images and corresponding camera/depth information from folders.

    Args:
        image_folder: Path to folder containing images
        camera_folder: Path to folder containing camera files (optional)
        depth_folder: Path to folder containing depth maps (optional)
        target_size: Target size for image resizing (default: 518)
        max_depth: Maximum valid depth value (default: 500)

    Returns:
        Tuple containing:
        - images: Stacked image tensors
        - extrinsics: Camera extrinsic matrices (world-to-camera)
        - intrinsics: Camera intrinsic matrices
        - depthmaps: Depth maps
        - masks: Valid depth masks
        - depth_indices: Indices of images with depth data
        - camera_indices: Indices of images with camera data
    """
    # Get all image files
    image_paths = sorted(glob.glob(os.path.join(image_folder, "*")))
    image_paths = [p for p in image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))]

    print(f"Found {len(image_paths)} images in {image_folder}")
    
    img_list = []
    extrinsics_list = []
    intrinsics_list = []
    depths_list = []
    masks_list = []
    depth_indices = []
    camera_indices = []

    for idx, img_path in enumerate(image_paths):
        basename = Path(img_path).stem
        img = Image.open(img_path)

        # Convert RGBA to RGB with white background
        if img.mode == "RGBA":
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            img = Image.alpha_composite(background, img)
        img = img.convert("RGB")
        width, height = img.size

        # Calculate resize parameters
        new_width = target_size
        new_height = round(height * (new_width / width) / 14) * 14
        scale_x = new_width / width
        scale_y = new_height / height

        # Resize image
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)

        # Calculate crop parameters if needed
        crop_start_y = 0
        final_height = new_height
        if new_height > target_size:
            crop_start_y = (new_height - target_size) // 2
            final_height = target_size
            # Crop image to target size
            img = img.crop((0, crop_start_y, new_width, crop_start_y + target_size))

        # Normalize image
        img = ImgNorm(img)
        img_list.append(img)

        # Initialize camera and depth data
        extrinsic = None
        intrinsic = None
        depthmap = None
        mask = None
        
        # Load depth map if available
        if depth_folder is not None:
            depth_candidates = [
                os.path.join(depth_folder, f"{basename}.npy"),
                os.path.join(depth_folder, f"{basename}.png"),
            ]
            for depth_path in depth_candidates:
                if os.path.exists(depth_path):
                    if depth_path.endswith('.npy'):
                        depthmap = np.load(depth_path).astype(np.float32)
                        depthmap[~np.isfinite(depthmap)] = 0  # invalid
                    elif depth_path.endswith('.png'):
                        depthmap = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
                        depthmap = depthmap.T
                        depthmap = np.nan_to_num(depthmap.astype(np.float32), 0.0)

                    # Filter invalid depth values
                    depthmap[depthmap > max_depth] = 0
                    depthmap[depthmap < 1e-5] = 0


        # Process depth map (follow the same resize/crop logic as image)
        if depthmap is not None:
            depth_indices.append(idx)
            # Resize depth map to match resized image dimensions (new_width x new_height)
            # cv2.resize expects (width, height) as the second parameter
            depthmap = cv2.resize(depthmap, (new_width, new_height), interpolation=cv2.INTER_NEAREST)

            # Crop depth map if needed (same crop as image)
            if new_height > target_size:
                depthmap = depthmap[crop_start_y : crop_start_y + target_size, :]

            mask = depthmap > 1e-5
        else:
            depthmap = np.zeros((final_height, new_width), dtype=np.float32)
            mask = np.zeros_like(depthmap, dtype=bool)

        depths_list.append(depthmap)
        masks_list.append(mask)

        # Load camera parameters if available
        if camera_folder is not None:
            camera_candidates = os.path.join(camera_folder, f"{basename}.txt")
            if os.path.exists(camera_candidates):
                extrinsic, intrinsic = load_camera_from_txt(camera_candidates)

        # Process camera parameters
        if extrinsic is not None and intrinsic is not None:
            camera_indices.append(idx)

            # Apply resize scaling to intrinsics
            intrinsic[0, 0] *= scale_x  # fx
            intrinsic[1, 1] *= scale_y  # fy
            intrinsic[0, 2] *= scale_x  # cx
            intrinsic[1, 2] *= scale_y  # cy

            # Apply crop adjustment to principal point y-coordinate
            if new_height > target_size:
                intrinsic[1, 2] -= crop_start_y  # cy

            # Convert camera-to-world to world-to-camera
            extrinsic = closed_form_inverse_se3(extrinsic[None])[0][:3]
        else:
            # Use zero matrices as placeholders
            extrinsic = np.zeros((3, 4), dtype=np.float32)
            intrinsic = np.zeros((3, 3), dtype=np.float32)

        extrinsics_list.append(extrinsic)
        intrinsics_list.append(intrinsic)
        

    print(f"\nSummary:")
    print(f"  Total images: {len(image_paths)}")
    print(f"  Images with camera: {len(camera_indices)} - indices: {camera_indices}")
    print(f"  Images with depth: {len(depth_indices)} - indices: {depth_indices}")

    images = torch.stack(img_list, dim=0)
    depthmaps = torch.from_numpy(np.array(depths_list))[None,...,None].float()
    masks = torch.from_numpy(np.array(masks_list))[None,...].float()
    extrinsics = torch.from_numpy(np.array(extrinsics_list))[None, ...].float()
    intrinsics = torch.from_numpy(np.array(intrinsics_list))[None, ...].float()

    return images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices

def load_camera_from_txt(camera_path: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load camera information from a text file.

    Expected file format:
    - First 3 lines: 3x4 extrinsic matrix (each line has 4 space/tab-separated values)
    - Next 3 lines: 3x3 intrinsic matrix (each line has 3 space/tab-separated values)

    Args:
        camera_path: Path to the camera text file

    Returns:
        Tuple of (extrinsic, intrinsic) or (None, None) if loading fails
    """
    try:
        with open(camera_path, 'r') as f:
            lines = f.readlines()

        # Remove empty lines and comments
        lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]

        if len(lines) < 6:
            print(f"  [WARN] Camera file has insufficient lines: {camera_path}")
            return None, None

        # Parse 3x4 extrinsic matrix
        extrinsic = []
        for i in range(3):
            values = [float(x) for x in lines[i].split()]
            if len(values) != 4:
                print(f"  [WARN] Invalid extrinsic matrix row {i}: {camera_path}")
                return None, None
            extrinsic.append(values)
        extrinsic = np.array(extrinsic, dtype=np.float32)

        # Parse 3x3 intrinsic matrix
        intrinsic = []
        for i in range(3, 6):
            values = [float(x) for x in lines[i].split()]
            if len(values) != 3:
                print(f"  [WARN] Invalid intrinsic matrix row {i-4}: {camera_path}")
                return None, None
            intrinsic.append(values)
        intrinsic = np.array(intrinsic, dtype=np.float32)

        return extrinsic, intrinsic
    except Exception as e:
        print(f"  [WARN] Failed to load camera from {camera_path}: {e}")
        return None, None
    
