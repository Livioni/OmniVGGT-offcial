import logging
import torch
import torch.nn as nn
from typing import Tuple, List

from omnivggt.layers import PatchEmbed
from omnivggt.layers.block import Block
from omnivggt.utils.pose_enc import extri_intri_to_pose_encoding
from torch.utils.checkpoint import checkpoint
from omnivggt.utils.geometry import closed_form_inverse_se3
from omnivggt.models.aggregator import Aggregator, slice_expand_and_flatten

logger = logging.getLogger(__name__)

_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]

class ZeroAggregator(Aggregator):
    def __init__(self, img_size=518, 
                 patch_size=14, 
                 embed_dim=1024, 
                 depth=24, 
                 num_heads=16, 
                 mlp_ratio=4, 
                 num_register_tokens=4, 
                 block_fn=Block, 
                 pose_hidden_dim=512,
                 qkv_bias=True, 
                 proj_bias=True, 
                 ffn_bias=True, 
                 patch_embed="dinov2_vitl14_reg", 
                 aa_order=["frame", "global"], 
                 aa_block_size=1, 
                 qk_norm=True, 
                 rope_freq=100, 
                 init_values=0.01,
                 enable_checkpoint=True):
        super().__init__(img_size, 
                         patch_size, 
                         embed_dim, 
                         depth, 
                         num_heads, 
                         mlp_ratio, 
                         num_register_tokens, 
                         block_fn,
                         qkv_bias, 
                         proj_bias, 
                         ffn_bias, 
                         patch_embed, 
                         aa_order, 
                         aa_block_size, 
                         qk_norm, 
                         rope_freq, 
                         init_values)
        
        
        self.patch_start_idx = 1 + num_register_tokens
        self.depth_placeholder = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        self.use_checkpoint = enable_checkpoint
        self.num_groups = self.aa_block_num + 1
        self.pose_embeddings   = nn.ModuleList()
        self.camera_adapters   = nn.ModuleList()

        for _ in range(self.num_groups):
            # pose_embedding
            pose_emb = nn.Linear(pose_hidden_dim, embed_dim)
            
            # camera adapter (zero init)
            cam_adapt = nn.Linear(embed_dim, embed_dim, bias=True)
            nn.init.zeros_(cam_adapt.weight)
            nn.init.zeros_(cam_adapt.bias)

            self.pose_embeddings.append(pose_emb)
            self.camera_adapters.append(cam_adapt)
            
        self.depth_patch_embed = PatchEmbed(img_size=img_size,
                                            patch_size=patch_size,
                                            in_chans=2,
                                            embed_dim=embed_dim)
        
    def _match_dtype(self, x, reference):
        return x.to(dtype=reference.dtype, device=reference.device)
    
    def normalize_extrinsics(self, extrinsics):
        B, S, _, _ = extrinsics.shape
        device = extrinsics.device
        extrinsics_homog = torch.cat(
            [
                extrinsics,
                torch.zeros((B, S, 1, 4), device=device),
            ],
            dim=-2,
        )
        extrinsics_homog[:, :, -1, -1] = 1.0
        first_cam_extrinsic_inv = closed_form_inverse_se3(extrinsics_homog[:, 0])
        new_extrinsics = torch.matmul(extrinsics_homog, first_cam_extrinsic_inv.unsqueeze(1))  # (B,N,4,4)
        
        if S > 1:
            cam_centers = new_extrinsics[:, :, :3, 3]  # (B, S, 3)
            ref_cam = cam_centers[:, 0:1, :]  # (B,1,3)
            rel_distances = torch.norm(cam_centers - ref_cam, dim=-1)[:,1:]  # (B, S)
            scale = rel_distances.mean(dim=1, keepdim=True).clamp(min=1e-6)  # (B, 1)
            new_extrinsics[:, :, :3, 3] /= scale.unsqueeze(-1)
        return new_extrinsics[:, :, :3]
    
    def normalize_depth(self, depth, mask, eps=1e-8):
        """
        depth: [B, V, H, W, 1]
        mask:  [B, V, H, W]
        """
        assert depth.shape[:4] == mask.shape, "mask and depth must have the same first four dimensions"

        B, V, H, W, _ = depth.shape
        depth_squeezed = depth.squeeze(-1)
        norm = torch.zeros_like(depth_squeezed)

        for b in range(B):
            valid = depth_squeezed[b][mask[b] > 0]
            if valid.numel() == 0:
                continue

            mean = valid.mean()
            norm_b = depth_squeezed[b] / (mean + eps)

            norm[b] = norm_b * mask[b]

        return norm.unsqueeze(-1)
    
    def forward(self, images: torch.Tensor, 
                extrinsics: torch.Tensor, 
                intrinsics: torch.Tensor,
                depth: torch.Tensor,
                mask: torch.Tensor,
                depth_gt_index: List[int],
                camera_gt_index: List[int]) -> Tuple[List[torch.Tensor], int]:
        B, S, C_in, H, W = images.shape
        
        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")

        # Normalize images and reshape for patch embed
        images = (images - self._resnet_mean) / self._resnet_std

        # Reshape to [B*S, C, H, W] for patch embedding
        images = images.view(B * S, C_in, H, W)
        patch_tokens = self.patch_embed(images)
            
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        K, P, C = patch_tokens.shape

        # Expand camera and register tokens to match batch size and sequence length
        camera_token = slice_expand_and_flatten(self.camera_token, B, S)
        register_token = slice_expand_and_flatten(self.register_token, B, S)
        
        if len(camera_gt_index) != 0:
            camera_gt_length = len(camera_gt_index)
            camera_idx_tensor = torch.tensor(camera_gt_index, device=depth.device)

            extrinsics_selected = torch.index_select(extrinsics, dim=1, index=camera_idx_tensor)
            intrinsics_selected = torch.index_select(intrinsics, dim=1, index=camera_idx_tensor)

            extrinsics_gt_normalized = self.normalize_extrinsics(extrinsics_selected)
            pose_encoding = extri_intri_to_pose_encoding(
                        extrinsics=extrinsics_gt_normalized,
                        intrinsics=intrinsics_selected,
                        image_size_hw=(H, W),
                        pose_encoding_type="absT_quaR_FoV",
            )
            gt_camera_token = self.pose_embeddings[0](pose_encoding).view(B * camera_gt_length, C).unsqueeze(1)
            
            device = depth.device
            camera_full = torch.zeros(K, 1, C, device=device, dtype=camera_token.dtype)

            camera_rows = (torch.arange(B, device=device).unsqueeze(1) * S + camera_idx_tensor.unsqueeze(0)).reshape(-1)
            camera_full[camera_rows] = gt_camera_token.to(dtype=camera_token.dtype)
            gt_camera_token = camera_full
        else:
            pose_encoding = None
            gt_camera_token = torch.zeros(K, 1, C, device=depth.device, dtype=camera_token.dtype)


        if len(depth_gt_index) != 0:
            depth_gt_length = len(depth_gt_index)
            idx_tensor = torch.tensor(depth_gt_index, device=depth.device)

            depth_selected = torch.index_select(depth, dim=1, index=idx_tensor)
            mask_selected = torch.index_select(mask, dim=1, index=idx_tensor)
            
            depth_gt_normalized = self.normalize_depth(depth_selected, mask_selected)
            
            depth_gt_normalized = depth_gt_normalized.view(B * depth_gt_length, 1, H, W)
            mask_selected = mask_selected.view(B * depth_gt_length, 1, H, W)
            
            depthmaps = torch.cat([depth_gt_normalized, mask_selected], dim=1)
            depthmaps = self._match_dtype(depthmaps, self.depth_patch_embed.proj.weight)
            gt_depth_token = self.depth_patch_embed(depthmaps)
            
            device = depth.device
            depth_full  = self.depth_placeholder.expand(K, P, C).clone()

            rows = (torch.arange(B, device=device).unsqueeze(1) * S + idx_tensor.unsqueeze(0)).reshape(-1)
            depth_full[rows]  = gt_depth_token.to(dtype = patch_tokens.dtype)     
            gt_depth_token  = depth_full                       
        else:
            gt_depth_token = self.depth_placeholder.expand(K, P, C)


        camera_token = camera_token + self.camera_adapters[0](gt_camera_token)
        patch_tokens = patch_tokens + gt_depth_token
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        pos = None
        if self.rope is not None:
            pos = self.position_getter(B * S, H // self.patch_size, W // self.patch_size, device=images.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * S, self.patch_start_idx, 2).to(images.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)

        # update P because we added special tokens
        P_old = P
        _, P, C = tokens.shape

        frame_idx = 0
        global_idx = 0
        output_list = []

        for index in range(self.aa_block_num):
            for attn_type in self.aa_order:
                if attn_type == "frame":
                    tokens, frame_idx, frame_intermediates = self._process_frame_attention(
                        tokens, B, S, P, C, frame_idx, pos=pos, index = index + 1, camera_gt_index = camera_gt_index,
                        pose_encoding=pose_encoding, register_shape = register_token.shape, P_old = P_old
                    )
                elif attn_type == "global":
                    tokens, global_idx, global_intermediates = self._process_global_attention(
                        tokens, B, S, P, C, global_idx, pos=pos
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            for i in range(len(frame_intermediates)):
                # concat frame and global intermediates, [B x S x P x 2C]
                concat_inter = torch.cat([frame_intermediates[i], global_intermediates[i]], dim=-1)
                output_list.append(concat_inter)

        del concat_inter
        del frame_intermediates
        del global_intermediates
        return output_list, self.patch_start_idx

    def _process_frame_attention(self, tokens, B, S, P, C, frame_idx, pos=None, index=None, camera_gt_index=None,
                         pose_encoding=None, register_shape = None, P_old = None):
        """
        Process frame attention blocks. We keep tokens in shape (B*S, P, C).
        """
        # If needed, reshape tokens or positions:
        if tokens.shape != (B * S, P, C):
            tokens = tokens.view(B, S, P, C).view(B * S, P, C)

        if pos is not None and pos.shape != (B * S, P, 2):
            pos = pos.view(B, S, P, 2).view(B * S, P, 2)

        intermediates = []

        K, _, _ = tokens.shape
        register_token = torch.zeros(register_shape, device=tokens.device, dtype=tokens.dtype).expand(B * S, -1, -1)
        if len(camera_gt_index) != 0:
            camera_gt_length = len(camera_gt_index)
            camera_idx_tensor = torch.tensor(camera_gt_index, device=tokens.device)
            gt_camera_token = self.pose_embeddings[index](pose_encoding).view(B * camera_gt_length, C).unsqueeze(1)
            camera_full = torch.zeros(K, 1, C, device=tokens.device, dtype=gt_camera_token.dtype)
            camera_rows = (torch.arange(B, device=tokens.device).unsqueeze(1) * S + camera_idx_tensor.unsqueeze(0)).reshape(-1)
            camera_full[camera_rows] = gt_camera_token.to(dtype=camera_full.dtype)
        else:
            camera_full = torch.zeros(K, 1, C, device=tokens.device, dtype=tokens.dtype)

        depth_injection= torch.zeros(K, P_old, C, device=tokens.device, dtype=tokens.dtype)

        camera_injection = self.camera_adapters[index](camera_full)
        injection_tokens = torch.cat([camera_injection, register_token, depth_injection], dim=1)

        # by default, self.aa_block_size=1, which processes one block at a time
        for _ in range(self.aa_block_size):
            blk = self.frame_blocks[frame_idx]
            if self.use_checkpoint and self.training:
                tokens = checkpoint(
                    lambda inp, p: blk(inp, pos=p,),
                    tokens,
                    pos,
                    use_reentrant=False
                )
            else:
                tokens = blk(tokens, pos=pos,)
            tokens = tokens + injection_tokens
            frame_idx += 1
            intermediates.append(tokens.view(B, S, P, C))

        return tokens, frame_idx, intermediates