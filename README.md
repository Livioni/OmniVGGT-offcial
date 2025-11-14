<div align="center">

<img src="assets/omnivggt.png" alt="OmniVGGT Logo" width="120"/>

<h1>OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer</h1>

<a href="https://arxiv.org/abs/2511.10560" target="blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-OmniVGGT-red" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2511.10560"><img src="https://img.shields.io/badge/arXiv-2511.10560-b31b1b" alt="arXiv"></a>
<a href="https://livioni.github.io/OmniVGGT-official"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>

---

Haosong Peng*, Hao Li*, Yalun Dai, Yushi Lan, Yihang Luo, Tianyu Qi, <br>
Zhengshen Zhang, Yufeng Zhan†, Junfei Zhang†, Wenchao Xu†, Ziwei Liu

 \* Equal Contribution, † Corresponding Author

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
# model to be released
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

- The *image_folder* contains all the images to be processed for reconstruction. The *camera_folder* and *depth_folder* are optional and may include any combination. For example, all the following combinations are ok.

<details>
<summary>📁 Click to see example folder structure combinations</summary>

```plaintext
example/infinigen
├── cameras
│   ├── 26_0_0001_0.txt
│   ├── 33_0_0001_0.txt
│   ├── 81_0_0001_0.txt
│   └── 91_0_0001_0.txt
├── depths
│   ├── 26_0_0001_0.npy
│   ├── 33_0_0001_0.npy
│   ├── 81_0_0001_0.npy
│   └── 91_0_0001_0.npy
└── images
    ├── 26_0_0001_0.png
    ├── 33_0_0001_0.png
    ├── 81_0_0001_0.png
    └── 91_0_0001_0.png
```

```plaintext
example/infinigen
├── cameras
│   ├── 26_0_0001_0.txt
│   └── 91_0_0001_0.txt
├── depths
│   ├── 33_0_0001_0.npy
│   └── 81_0_0001_0.npy
└── images
    ├── 26_0_0001_0.png
    ├── 33_0_0001_0.png
    ├── 81_0_0001_0.png
    └── 91_0_0001_0.png
```

```plaintext
example/infinigen
├── cameras
│   ├── 26_0_0001_0.txt
│   └── 33_0_0001_0.txt
├── depths
│   └── 91_0_0001_0.npy
└── images
    ├── 26_0_0001_0.png
    ├── 33_0_0001_0.png
    ├── 81_0_0001_0.png
    └── 91_0_0001_0.png
```

</details>

- If one or more images have auxiliary camera information, please ensure that the first image always includes camera information.
- Camera poses and intrinsics are provided in **.txt** files. Please refer to [frame-000002.txt](example/office/cameras/frame-000002.txt) for specific examples. Depth maps can be loaded from either **.png** or **.npy** files.
- Camera poses are expected to follow the OpenCV `camera-from-world` convention, Depth maps should be aligned with their corresponding camera poses.

## 📸 Example

### Comparison: Without vs. With Camera Parameters

<div align="center">
  <img src="assets/left.png" alt="Without Camera" width="400"/>
  <img src="assets/right.png" alt="With Camera" width="400"/>
</div>

**Left**: Results without auxiliary camera parameters

```bash
python inference.py --image_folder example/office/images
```

**Right**: Results with auxiliary camera parameters

```bash
python inference.py --image_folder example/office/images --camera_folder example/office/cameras
```

## 📝 To-Do List

- [X] Release project paper.
- [ ] Release pretrained models.
- [ ] Release training code.

## 🤝 Citation

If you use this code in your research, please cite:

```bibtex
{omnivggt2025,
  title={OmniVGGT: Omni-Modality Driven Visual Geometry Grounded Transformer},
  author={Haosong Peng and Hao Li and Yalun Dai and Yushi Lan and Yihang Luo and Tianyu Qi and Zhengshen Zhang and Yufeng Zhan and Junfei Zhang and Wenchao Xu and Ziwei Liu}
  journal={arXiv preprint arXiv:2511.10560},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built upon [VGGT](https://github.com/facebookresearch/vggt) by Meta AI
- Uses [viser](https://github.com/nerfstudio-project/viser) for 3D visualization
