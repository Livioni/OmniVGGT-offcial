<div align="center">

<img src="assets/omnivggt.png" alt="OmniVGGT Logo" width="120"/>

<h1>OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer</h1>

<a href="https://arxiv.org/abs/2510.22706" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-OmniVGGT-red" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2510.22706"><img src="https://img.shields.io/badge/arXiv-2510.22706-b31b1b" alt="arXiv"></a>
<a href="https://livioni.github.io/OmniVGGT-offcial"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

---

Haosong Peng `<sup>`*`</sup>`, Hao Li `<sup>`* `</sup>`, Yalun Dai, Yushi Lan, Yihang Luo, Tianyu Qi,
Zhengshen Zhang, Yufeng Zhan `<sup>`†`</sup>`, Junfei Zhang `<sup>`†`</sup>`, Wenchao Xu `<sup>`†`</sup>`, Ziwei Liu

`<sup>`*`</sup>`Equal Contribution, `<sup>`†`</sup>`Corresponding Author

</div>

<div align="center">
  <img src="assets/teaser.png" alt="OmniVGGT Overview" width="800"/>
</div>

## 🔍 Overview

OmniVGGT is a spatial foundation model that can effectively benefit from an arbitrary number of auxiliary geometric modalities (depth, camera intrinsics and pose) to obtain high-quality 3D geometric results. Experimental results show that OmniVGGT achieves state-of-the-art performance across various downstream tasks and further improves performance on robot manipulation tasks.

## 🔧 Installation

### Setup Environment

```bash
conda create -n omnivggt python=3.10

conda activate omnivggt

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128

pip install -r requirements.txt
```

## 📦 Model Weights

Download the pretrained model weights:

- **OmniVGGT Model**: [Download Link] (To be provided)

Place the downloaded weights in the `checkpoints/` directory.

## 🚀 Quick Start

You can use OmniVGGT directly in your Python code:

```python
import torch
from omnivggt.models.omnivggt import OmniVGGT
from omnivggt.utils.pose_enc import pose_encoding_to_extri_intri
from visual_util import load_images_and_cameras

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the model
model = OmniVGGT().to(device)
from safetensors.torch import load_file
state_dict = load_file("checkpoints/from132_136k.safetensors")
model.load_state_dict(state_dict, strict=True)
model.eval()

# Load and preprocess images
images, extrinsics, intrinsics, depthmaps, masks, depth_indices, camera_indices = \
    load_images_and_cameras(
        image_folder="example/office/images/",
        camera_folder=None,  # Optional
        depth_folder=None,   # Optional
        target_size=518
    )

# Prepare inputs
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
with torch.no_grad():
    predictions = model(**inputs)

# Convert pose encoding to camera matrices
extrinsic, intrinsic = pose_encoding_to_extri_intri(
    predictions["pose_enc"],
    images.shape[-2:]
)
predictions["extrinsic"] = extrinsic
predictions["intrinsic"] = intrinsic

# Access the results
depth_maps = predictions["depth"]           # (B, S, H, W, 1)
depth_conf = predictions["depth_conf"]      # (B, S, H, W)
pred_extrinsics = predictions["extrinsic"]  # (B, S, 3, 4)
pred_intrinsics = predictions["intrinsic"]  # (B, S, 3, 3)
```

### Advanced Options

With Anuxiliary Camera and Depth:

If you have Anuxiliary truth camera parameters and/or depth maps:

```bash
python inference.py \
    --image_folder example/office/images/ \ 
    --camera_folder example/office/cameras/ \ #optional
    --depth_folder example/office/depths/ #optional
```

## 📊 Input Description

- The *image_folder* contains all the images to be processed for reconstruction. The *camera_folder* and *depth_folder* are optional and may include poses and depth maps for any subset of the images.
  For example, all the following combinations are ok.

<table>
<tr>
<td>

- If one or more images have auxiliary camera information, please ensure that the first image always includes camera information.
- Camera poses and intrinsics are provided in **.txt** files. Please refer to [frame-000002.txt](example/office/cameras/frame-000002.txt) for specific examples. Depth maps can be loaded from either **.png** or **.npy** files.
- Camera poses are expected to follow the OpenCV `camera-from-world` convention, Depth maps should be aligned with their corresponding camera poses.

## 

## 📝 To-Do List

- [ ] Release project paper.
- [ ] Release pretrained models.
- [ ] Release training code.

## 🤝 Citation

If you use this code in your research, please cite:

```bibtex
@article{li2025iggt,
  title={{IGGT}: {INSTANCE-GROUNDED} {GEOMETRY} {TRANSFORMER} for {SEMANTIC} {3D} {RECONSTRUCTION}},
  author={Li, Hao and Zou, Zhengyu and Liu, Fangfu and Zhang, Xuanyang and Hong, Fangzhou and Cao, Yukang and Lan, Yushi and Zhang, Manyuan and Yu, Gang and Zhang, Dingwen and Liu, Ziwei},
  journal={arXiv preprint arXiv:24XX.XXXXX},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built upon [VGGT](https://github.com/facebookresearch/vggt) by Meta AI
- Uses [viser](https://github.com/nerfstudio-project/viser) for 3D visualization
