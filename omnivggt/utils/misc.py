# --------------------------------------------------------
# utilitary functions for DUSt3R
# --------------------------------------------------------
import torch
import cv2
import numpy as np
from omnivggt.utils.vo_eval import save_trajectory_tum_format
from typing import Optional, Tuple
from PIL import Image

def closed_form_inverse_se3(se3, R=None, T=None):
    """
    Compute the inverse of each 4x4 (or 3x4) SE3 matrix in a batch.

    If `R` and `T` are provided, they must correspond to the rotation and translation
    components of `se3`. Otherwise, they will be extracted from `se3`.

    Args:
        se3: Nx4x4 or Nx3x4 array or tensor of SE3 matrices.
        R (optional): Nx3x3 array or tensor of rotation matrices.
        T (optional): Nx3x1 array or tensor of translation vectors.

    Returns:
        Inverted SE3 matrices with the same type and device as `se3`.

    Shapes:
        se3: (N, 4, 4)
        R: (N, 3, 3)
        T: (N, 3, 1)
    """
    # Check if se3 is a numpy array or a torch tensor
    is_numpy = isinstance(se3, np.ndarray)

    # Validate shapes
    if se3.shape[-2:] != (4, 4) and se3.shape[-2:] != (3, 4):
        raise ValueError(f"se3 must be of shape (N,4,4), got {se3.shape}.")

    # Extract R and T if not provided
    if R is None:
        R = se3[:, :3, :3]  # (N,3,3)
    if T is None:
        T = se3[:, :3, 3:]  # (N,3,1)

    # Transpose R
    if is_numpy:
        # Compute the transpose of the rotation for NumPy
        R_transposed = np.transpose(R, (0, 2, 1))
        # -R^T t for NumPy
        top_right = -np.matmul(R_transposed, T)
        inverted_matrix = np.tile(np.eye(4), (len(R), 1, 1))
    else:
        R_transposed = R.transpose(1, 2)  # (N,3,3)
        top_right = -torch.bmm(R_transposed, T)  # (N,3,1)
        inverted_matrix = torch.eye(4, 4)[None].repeat(len(R), 1, 1)
        inverted_matrix = inverted_matrix.to(R.dtype).to(R.device)

    inverted_matrix[:, :3, :3] = R_transposed
    inverted_matrix[:, :3, 3:] = top_right

    return inverted_matrix

def get_stride_distribution(strides, dist_type='uniform'):

    # input strides sorted by descreasing order by default
    
    if dist_type == 'uniform':
        dist = np.ones(len(strides)) / len(strides)
    elif dist_type == 'exponential':
        lambda_param = 1.0
        dist = np.exp(-lambda_param * np.arange(len(strides)))
    elif dist_type.startswith('linear'): # e.g., linear_1_2
        try:
            start, end = map(float, dist_type.split('_')[1:])
            dist = np.linspace(start, end, len(strides))
        except ValueError:
            raise ValueError(f'Invalid linear distribution format: {dist_type}')
    else:
        raise ValueError('Unknown distribution type %s' % dist_type)

    # normalize to sum to 1
    return dist / np.sum(dist)


def fill_default_args(kwargs, func):
    import inspect  # a bit hacky but it works reliably
    signature = inspect.signature(func)

    for k, v in signature.parameters.items():
        if v.default is inspect.Parameter.empty:
            continue
        kwargs.setdefault(k, v.default)

    return kwargs


def freeze_all_params(modules):
    for module in modules:
        try:
            for n, param in module.named_parameters():
                param.requires_grad = False
        except AttributeError:
            # module is directly a parameter
            module.requires_grad = False


def is_symmetrized(gt1, gt2):
    x = gt1['instance']
    y = gt2['instance']
    if len(x) == len(y) and len(x) == 1:
        return False  # special case of batchsize 1
    ok = True
    for i in range(0, len(x), 2):
        ok = ok and (x[i] == y[i + 1]) and (x[i + 1] == y[i])
    return ok


def flip(tensor):
    """ flip so that tensor[0::2] <=> tensor[1::2] """
    return torch.stack((tensor[1::2], tensor[0::2]), dim=1).flatten(0, 1)


def interleave(tensor1, tensor2):
    res1 = torch.stack((tensor1, tensor2), dim=1).flatten(0, 1)
    res2 = torch.stack((tensor2, tensor1), dim=1).flatten(0, 1)
    return res1, res2


def transpose_to_landscape(head, activate=True):
    """ Predict in the correct aspect-ratio,
        then transpose the result in landscape 
        and stack everything back together.
    """
    def wrapper_no(decout, true_shape):
        B = len(true_shape)
        assert true_shape[0:1].allclose(true_shape), 'true_shape must be all identical'
        H, W = true_shape[0].cpu().tolist()
        res = head(decout, (H, W))
        return res

    def wrapper_yes(decout, true_shape):
        B = len(true_shape)
        # by definition, the batch is in landscape mode so W >= H
        H, W = int(true_shape.min()), int(true_shape.max())

        height, width = true_shape.T
        is_landscape = (width >= height)
        is_portrait = ~is_landscape

        # true_shape = true_shape.cpu()
        if is_landscape.all():
            return head(decout, (H, W))
        if is_portrait.all():
            return transposed(head(decout, (W, H)))

        # batch is a mix of both portraint & landscape
        def selout(ar): return [d[ar] for d in decout]
        l_result = head(selout(is_landscape), (H, W))
        p_result = transposed(head(selout(is_portrait), (W, H)))

        # allocate full result
        result = {}
        for k in l_result | p_result:
            x = l_result[k].new(B, *l_result[k].shape[1:])
            x[is_landscape] = l_result[k]
            x[is_portrait] = p_result[k]
            result[k] = x

        return result

    return wrapper_yes if activate else wrapper_no


def transposed(dic):
    return {k: v.swapaxes(1, 2) for k, v in dic.items()}


def invalid_to_nans(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float('nan')
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr


def invalid_to_zeros(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = 0
        nnz = valid_mask.view(len(valid_mask), -1).sum(1)
    else:
        nnz = arr.numel() // len(arr) if len(arr) else 0  # number of point per image
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr, nnz

def save_tum_poses(traj, path):
    # traj = self.get_tum_poses()
    save_trajectory_tum_format(traj, path)
    return traj[0] # return the poses

def save_focals(focals, path):
    # convert focal to txt
    # focals = self.get_focals()
    np.savetxt(path, focals.detach().cpu().numpy(), fmt='%.6f')
    return focals

def save_intrinsics(K_raw, path):
    # K_raw = self.get_intrinsics()
    K = K_raw.reshape(-1, 9)
    np.savetxt(path, K.detach().cpu().numpy(), fmt='%.6f')
    return K_raw

def save_conf_maps(conf, path):
    # conf = self.get_conf()
    for i, c in enumerate(conf):
        np.save(f'{path}/conf_{i}.npy', c.detach().cpu().numpy())
    return conf

def save_rgb_imgs(imgs, path):
    # imgs = self.imgs
    for i, img in enumerate(imgs):
        # convert from rgb to bgr
        img = img[..., ::-1]
        cv2.imwrite(f'{path}/frame_{i:04d}.png', img*255)
    return imgs

def save_dynamic_masks(dynamic_masks, path):
    # dynamic_masks = self.dynamic_masks
    for i, dynamic_mask in enumerate(dynamic_masks):
        cv2.imwrite(f'{path}/dynamic_mask_{i}.png', (dynamic_mask * 255).detach().cpu().numpy().astype(np.uint8))
    return dynamic_masks

def save_depth_maps(depth_maps, path):
    images = []
    for i, depth_map in enumerate(depth_maps):
        depth_map_colored = cv2.applyColorMap((depth_map * 255).detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
        img_path = f'{path}/frame_{(i):04d}.png'
        cv2.imwrite(img_path, depth_map_colored)
        images.append(Image.open(img_path))
        # Save npy file
        np.save(f'{path}/frame_{(i):04d}.npy', depth_map.detach().cpu().numpy())
    
    # Save gif using Pillow
    images[0].save(f'{path}/_depth_maps.gif', save_all=True, append_images=images[1:], duration=100, loop=0)
    return depth_maps

def to_cpu(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    if isinstance(x, list):
        return [to_cpu(xx) for xx in x]


def select_first_batch(inputs, weight_dtype=None):
    """
    Remove batch dimension from all tensors in the inputs dictionary and optionally change data type.
    
    Args:
        inputs (dict): Dictionary containing tensors with batch dimension.
        keys (list, optional): Keys to process. If None, processes all tensors.
        weight_dtype (torch.dtype, optional): If specified, converts all tensors to this data type.
    
    Returns:
        dict: Dictionary with batch dimension removed and data type converted.
    """
    new_dicts = {}
    keys = ["pose_enc", "depth", "world_points", "images", "extrinsic", "intrinsic", "world_points_from_depth", 'depth_conf', 'world_points_conf']
    for key, value in inputs.items():
        if isinstance(value, torch.Tensor) and (keys is None or key in keys):
            try:
                value = value[:1, ...].detach()
            except:
                value = value[:1, ...]
            
            if weight_dtype:
                value = value.to(weight_dtype)
        elif isinstance(value, list) and key == "pose_enc":
            value = value[-1]
            value = value[:1, ...].detach()
        
        new_dicts[key] = value
    
    return new_dicts


def save_images_from_tensor(tensor, save_dir="frames", prefix="frame"):
    """
    Save tensor frames as images.

    Args:
        tensor (torch.Tensor): Image tensor with shape (1, 4, H, W, 3), values in [0, 1] or [0, 255].
        save_dir (str): Directory to save images.
        prefix (str): Filename prefix.
    """
    import os

    os.makedirs(save_dir, exist_ok=True)

    tensor = tensor.squeeze(0)

    for i, img_tensor in enumerate(tensor):
        img_array = img_tensor.permute(1,2,0).cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)

        img = Image.fromarray(img_array)

        img.save(f"{save_dir}/{prefix}_{i}.png")
        print(f"Saved: {save_dir}/{prefix}_{i}.png")

def check_valid_tensor(input_tensor: Optional[torch.Tensor], name: str = "tensor") -> None:
    """
    Check if a tensor contains NaN or Inf values and log a warning if found.
    
    Args:
        input_tensor: The tensor to check
        name: Name of the tensor for logging purposes
    """
    if input_tensor is not None:
        if torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any():
            logging.warning(f"NaN or Inf found in tensor: {name}")

def normalize_camera_extrinsics_and_points_batch(
    extrinsics: torch.Tensor,
    cam_points: Optional[torch.Tensor] = None,
    world_points: Optional[torch.Tensor] = None,
    depths: Optional[torch.Tensor] = None,
    scale_by_points: bool = True,
    point_masks: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Normalize camera extrinsics and corresponding 3D points.
    
    This function transforms the coordinate system to be centered at the first camera
    and optionally scales the scene to have unit average distance.
    
    Args:
        extrinsics: Camera extrinsic matrices of shape (B, S, 3, 4)
        cam_points: 3D points in camera coordinates of shape (B, S, H, W, 3) or (*,3)
        world_points: 3D points in world coordinates of shape (B, S, H, W, 3) or (*,3)
        depths: Depth maps of shape (B, S, H, W)
        scale_by_points: Whether to normalize the scale based on point distances
        point_masks: Boolean masks for valid points of shape (B, S, H, W)
    
    Returns:
        Tuple containing:
        - Normalized camera extrinsics of shape (B, S, 3, 4)
        - Normalized camera points (same shape as input cam_points)
        - Normalized world points (same shape as input world_points)
        - Normalized depths (same shape as input depths)
    """
    # Validate inputs
    check_valid_tensor(extrinsics, "extrinsics")
    check_valid_tensor(cam_points, "cam_points")
    check_valid_tensor(world_points, "world_points")
    check_valid_tensor(depths, "depths")


    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    # assert device == torch.device("cpu")
    depths = depths.squeeze(-1)

    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)


    if world_points is not None:
        # since we are transforming the world points to the first camera's coordinate system
        # we directly use the cam_from_world extrinsic matrix of the first camera
        # instead of using the inverse of the first camera's extrinsic matrix
        R = extrinsics[:, 0, :3, :3]
        t = extrinsics[:, 0, :3, 3]
        new_world_points = (world_points @ R.transpose(-1, -2).unsqueeze(1).unsqueeze(2)) + t.unsqueeze(1).unsqueeze(2).unsqueeze(3)
    else:
        new_world_points = None


    if scale_by_points:
        if cam_points is not None:
            new_cam_points = cam_points.clone()
        else:
            new_cam_points = None
        new_depths = depths.clone()

        dist = new_world_points.norm(dim=-1)
        dist_sum = (dist * point_masks).sum(dim=[1,2,3])
        valid_count = point_masks.sum(dim=[1,2,3])
        avg_scale = (dist_sum / (valid_count + 1e-3)).clamp(min=1e-6, max=1e6)


        new_world_points = new_world_points / avg_scale.view(-1, 1, 1, 1, 1)
        new_extrinsics[:, :, :3, 3] = new_extrinsics[:, :, :3, 3] / avg_scale.view(-1, 1, 1)
        if depths is not None:
            new_depths = new_depths / avg_scale.view(-1, 1, 1, 1)
        if cam_points is not None:
            new_cam_points = new_cam_points / avg_scale.view(-1, 1, 1, 1, 1)
    else:
        return new_extrinsics[:, :, :3], cam_points, new_world_points, depths

    new_extrinsics = new_extrinsics[:, :, :3] # 4x4 -> 3x4
    new_extrinsics = check_and_fix_inf_nan(new_extrinsics, "new_extrinsics")
    if new_cam_points is not None:
        new_cam_points = check_and_fix_inf_nan(new_cam_points, "new_cam_points", hard_max=None)
    if new_world_points is not None:
        new_world_points = check_and_fix_inf_nan(new_world_points, "new_world_points")
    new_depths = check_and_fix_inf_nan(new_depths, "new_depths")


    return new_extrinsics, new_cam_points, new_world_points, new_depths


def gt_injection(
    extrinsics,
    world_points=None,
    depths=None,
    point_masks=None,
):
    # Note this assumes we use cpu
    # extrinsics: (B, S, 3, 4)
    # world_points: (B, S, H, W, 3) or (*,3)
    # cam_points: same shape as world_points or something consistent
    # point_masks: (B, S, H, W) boolean mask if provided
        
    B, S, _, _ = extrinsics.shape
    device = extrinsics.device
    dtype = extrinsics.dtype


    # Convert extrinsics to homogeneous form: (B, N,4,4)
    extrinsics_homog = torch.cat(
        [
            extrinsics,
            torch.zeros((B, S, 1, 4), device=device),
        ],
        dim=-2,
    )
    extrinsics_homog[:, :, -1, -1] = 1.0

    # first_cam_extrinsic_inv, the inverse of the first camera's extrinsic matrix
    # which can be also viewed as the cam_to_world extrinsic matrix
    first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
    # new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv)
    new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)

    return new_extrinsics
