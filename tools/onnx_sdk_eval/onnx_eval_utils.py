import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# from mmcv.runner import BaseModule
# class SinePositionalEncoding3D(BaseModule):
#     """Position encoding with sine and cosine functions.
#     See `End-to-End Object Detection with Transformers
#     <https://arxiv.org/pdf/2005.12872>`_ for details.
#     Args:
#         num_feats (int): The feature dimension for each position
#             along x-axis or y-axis. Note the final returned dimension
#             for each position is 2 times of this value.
#         temperature (int, optional): The temperature used for scaling
#             the position embedding. Defaults to 10000.
#         normalize (bool, optional): Whether to normalize the position
#             embedding. Defaults to False.
#         scale (float, optional): A scale factor that scales the position
#             embedding. The scale will be used only when `normalize` is True.
#             Defaults to 2*pi.
#         eps (float, optional): A value added to the denominator for
#             numerical stability. Defaults to 1e-6.
#         offset (float): offset add to embed when do the normalization.
#             Defaults to 0.
#         init_cfg (dict or list[dict], optional): Initialization config dict.
#             Default: None
#     """

#     def __init__(self,
#                  num_feats,
#                  temperature=10000,
#                  normalize=False,
#                  scale=2 * math.pi,
#                  eps=1e-6,
#                  offset=0.,
#                  init_cfg=None):
#         super(SinePositionalEncoding3D, self).__init__(init_cfg)
#         if normalize:
#             assert isinstance(scale, (float, int)), 'when normalize is set,' \
#                 'scale should be provided and in float or int type, ' \
#                 f'found {type(scale)}'
#         self.num_feats = num_feats
#         self.temperature = temperature
#         self.normalize = normalize
#         self.scale = scale
#         self.eps = eps
#         self.offset = offset

#     def forward(self, mask):
#         """Forward function for `SinePositionalEncoding`.
#         Args:
#             mask (Tensor): ByteTensor mask. Non-zero values representing
#                 ignored positions, while zero values means valid positions
#                 for this image. Shape [bs, h, w].
#         Returns:
#             pos (Tensor): Returned position embedding with shape
#                 [bs, num_feats*2, h, w].
#         """
#         # For convenience of exporting to ONNX, it's required to convert
#         # `masks` from bool to int.
#         mask = mask.to(torch.int)
#         not_mask = 1 - mask  # logical_not
#         n_embed = not_mask.cumsum(1, dtype=torch.float32)
#         y_embed = not_mask.cumsum(2, dtype=torch.float32)
#         x_embed = not_mask.cumsum(3, dtype=torch.float32)
#         if self.normalize:
#             n_embed = (n_embed + self.offset) / \
#                       (n_embed[:, -1:, :, :] + self.eps) * self.scale
#             y_embed = (y_embed + self.offset) / \
#                       (y_embed[:, :, -1:, :] + self.eps) * self.scale
#             x_embed = (x_embed + self.offset) / \
#                       (x_embed[:, :, :, -1:] + self.eps) * self.scale
#         dim_t = torch.arange(
#             self.num_feats, dtype=torch.float32, device=mask.device)
#         dim_t = self.temperature**(2 * (dim_t // 2) / self.num_feats)
#         pos_n = n_embed[:, :, :, :, None] / dim_t
#         pos_x = x_embed[:, :, :, :, None] / dim_t
#         pos_y = y_embed[:, :, :, :, None] / dim_t
#         # use `view` instead of `flatten` for dynamically exporting to ONNX
#         B, N, H, W = mask.size()
#         pos_n = torch.stack(
#             (pos_n[:, :, :, :, 0::2].sin(), pos_n[:, :, :, :, 1::2].cos()),
#             dim=4).view(B, N, H, W, -1)
#         pos_x = torch.stack(
#             (pos_x[:, :, :, :, 0::2].sin(), pos_x[:, :, :, :, 1::2].cos()),
#             dim=4).view(B, N, H, W, -1)
#         pos_y = torch.stack(
#             (pos_y[:, :, :, :, 0::2].sin(), pos_y[:, :, :, :, 1::2].cos()),
#             dim=4).view(B, N, H, W, -1)
#         pos = torch.cat((pos_n, pos_y, pos_x), dim=4).permute(0, 1, 4, 2, 3)
#         return pos

#     def __repr__(self):
#         """str: a string that describes the module"""
#         repr_str = self.__class__.__name__
#         repr_str += f'(num_feats={self.num_feats}, '
#         repr_str += f'temperature={self.temperature}, '
#         repr_str += f'normalize={self.normalize}, '
#         repr_str += f'scale={self.scale}, '
#         repr_str += f'eps={self.eps})'
#         return repr_str


def pos2posemb3d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_z = pos[..., 2, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x, pos_z), dim=-1)
    return posemb

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)
    

class PE(torch.nn.Module):
    def __init__(self, position_level=0):
        super().__init__()
        
        self.export_onnx = True
        
        self.position_level = position_level
        self.depth_num = 64
        self.position_dim = 3 * self.depth_num
        self.embed_dims = 256
        self.position_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]
        self.depth_start = 1
        self.LID = True


        self.position_encoder = nn.Sequential(
            nn.Conv2d(self.position_dim, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )

    def forward(self, img_feats, img_metas, masks):
        
        eps = 1e-5
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        
        B, N, C, H, W = img_feats[self.position_level].shape
        coords_h = torch.arange(H, device=img_feats[0].device).float() * pad_h / H
        coords_w = torch.arange(W, device=img_feats[0].device).float() * pad_w / W

        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1, device=img_feats[0].device).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        D = coords_d.shape[0]
        coords = torch.stack(torch.meshgrid([coords_w, coords_h, coords_d])).permute(1, 2, 3, 0) # W, H, D, 3
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        if not self.export_onnx:
            coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)
        else:
            coords[..., :2] = coords[..., :2] * coords[..., 2:3]

        img2lidars = []
        for img_meta in img_metas:
            img2lidar = []
            for i in range(len(img_meta['lidar2img'])):
                img2lidar.append(np.linalg.inv(img_meta['lidar2img'][i]))
            img2lidars.append(np.asarray(img2lidar))
        img2lidars = np.asarray(img2lidars)
        img2lidars = coords.new_tensor(img2lidars) # (B, N, 4, 4)

        coords = coords.view(1, 1, W, H, D, 4, 1).repeat(B, N, 1, 1, 1, 1, 1)
        img2lidars = img2lidars.view(B, N, 1, 1, 1, 4, 4).repeat(1, 1, W, H, D, 1, 1)
        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:1] = (coords3d[..., 0:1] - self.position_range[0]) / (self.position_range[3] - self.position_range[0])
        coords3d[..., 1:2] = (coords3d[..., 1:2] - self.position_range[1]) / (self.position_range[4] - self.position_range[1])
        coords3d[..., 2:3] = (coords3d[..., 2:3] - self.position_range[2]) / (self.position_range[5] - self.position_range[2])

        coords_mask = (coords3d > 1.0) | (coords3d < 0.0) 
        coords_mask = coords_mask.flatten(-2).sum(-1) > (D * 0.5)
        coords_mask = masks | coords_mask.permute(0, 1, 3, 2)
        coords3d = coords3d.permute(0, 1, 4, 5, 3, 2).contiguous().view(B*N, -1, H, W)
        coords3d = inverse_sigmoid(coords3d)

        coords_position_embeding = self.position_encoder(coords3d)
        
        return coords_position_embeding.view(B, N, self.embed_dims, H, W), coords_mask


def load_weights(model, weight_path, device='cuda:0'):
    sd = model.state_dict()
    model_weights = np.load(weight_path, allow_pickle=True)['arr_0'][()]
    for k, v in model_weights.items():
        # sd[k].copy_(torch.tensor(v).to(device))
        sd[k].copy_(torch.tensor(v))


class QueryEmbed(torch.nn.Module):
    def __init__(self,
                 embed_dims=256,
                 num_query=900,
                 state_dict_path=None):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_query = num_query
        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.query_embedding#.eval()
        load_weights(self.query_embedding, state_dict_path)
    
    def forward(self, x):
        return self.query_embedding(x)


class Coords3DPE(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 embed_dims=256,
                 input_norm=False,
                 state_dict_paths=None,
                 position_level=0,
                 **kwargs
                 ):
        super().__init__()
        self.embed_dims = embed_dims
        self.position_level = position_level
        self.in_channels = in_channels

        self.masks = Mask(position_level=self.position_level)
        self.position_embeding = PE(position_level=position_level)
        self.fpe = SELayer(self.embed_dims)
        self.input_proj = nn.Conv2d(
                self.in_channels, self.embed_dims, kernel_size=1)
        self.adapt_pos3d = nn.Sequential(
                nn.Conv2d(self.embed_dims*3//2, self.embed_dims*4, kernel_size=1, stride=1, padding=0),
                nn.ReLU(),
                nn.Conv2d(self.embed_dims*4, self.embed_dims, kernel_size=1, stride=1, padding=0),
            )
        
        from projects.mmdet3d_plugin.models.utils.positional_encoding import SinePositionalEncoding3D
        self.positional_encoding = SinePositionalEncoding3D(num_feats=128,
                                                            normalize=True)
        self.batch_norms = None
        if input_norm:
            self.batch_norms = nn.ModuleList([
                nn.BatchNorm1d(num_features=embed_dims),
                nn.BatchNorm1d(num_features=embed_dims)
                ])
            for i, batch_norm in enumerate(self.batch_norms):
                load_weights(batch_norm, state_dict_paths["batch_norms"][i])
        
        load_weights(self.position_embeding.position_encoder, state_dict_paths["position_encoder"])
        load_weights(self.fpe, state_dict_paths["fpe"])
        load_weights(self.input_proj, state_dict_paths["input_proj"])
        load_weights(self.adapt_pos3d, state_dict_paths["adapt_pos3d"])

    def forward(self, img_metas, mlvl_feats=None):
        x = mlvl_feats[self.position_level]
        batch_size, num_cams = x.size(0), x.size(1)

        masks = self.masks(img_metas, mlvl_feats)

        x = self.input_proj(x.flatten(0,1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)

        coords_position_embeding, _ = self.position_embeding(mlvl_feats, img_metas, masks)
        coords_position_embeding = self.fpe(coords_position_embeding.flatten(0,1), x.flatten(0,1)).view(x.size())
        pos_embed = coords_position_embeding

        sin_embed = self.positional_encoding(masks)
        sin_embed = self.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
        pos_embed = pos_embed + sin_embed

        if self.batch_norms is not None:
            bs, n, c, h, w = x.shape
            memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
            pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
            pos_embed = self.batch_norms[0].to(pos_embed.device)(pos_embed.permute(1, 2, 0)).permute(2, 0, 1)
            memory = self.batch_norms[1].to(memory.device)(memory.permute(1, 2, 0)).permute(2, 0, 1)
            return pos_embed, memory
        else:
            return pos_embed, x
    
class Mask(torch.nn.Module):
    def __init__(self, position_level=0):
        super().__init__()
        self.position_level = position_level

    def forward(self, img_metas, mlvl_feats=None):
        x = mlvl_feats[self.position_level]
        batch_size, num_cams = x.size(0), x.size(1)

        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))

        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0

        return masks
    

class RegLayer(nn.Module):
    def __init__(self,  embed_dims=256, 
                        shared_reg_fcs=2, 
                        group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
                        act_layer=nn.ReLU, 
                        drop=0.0):
        super().__init__()

        reg_branch = []
        for _ in range(shared_reg_fcs):
            # reg_branch.append(Linear(embed_dims, embed_dims))
            reg_branch.append(nn.Linear(embed_dims, embed_dims))
            reg_branch.append(act_layer())
            reg_branch.append(nn.Dropout(drop))
        self.reg_branch = nn.Sequential(*reg_branch)

        self.task_heads = nn.ModuleList()
        for reg_dim in group_reg_dims:
            task_head = nn.Sequential(
                # Linear(embed_dims, embed_dims),
                nn.Linear(embed_dims, embed_dims),
                act_layer(),
                # Linear(embed_dims, reg_dim)
                nn.Linear(embed_dims, reg_dim)
            )
            self.task_heads.append(task_head)

    def forward(self, x):
        reg_feat = self.reg_branch(x)
        outs = []
        for task_head in self.task_heads:
            out = task_head(reg_feat.clone())
            outs.append(out)
        outs = torch.cat(outs, -1)
        return outs

import copy
class PredictionLayer(torch.nn.Module):
    def __init__(self,
                 cls_out_channels,
                 num_reg_fcs=2,
                 group_reg_dims=(2, 1, 3, 2, 2),  # xy, z, size, rot, velo
                 embed_dims=None,
                 cls_branches_state_dict_path=None,
                 reg_branches_state_dict_path=None,
                 num_pred=6
                 ):
        super().__init__()

        self.num_reg_fcs = num_reg_fcs
        self.embed_dims = embed_dims
        self.cls_out_channels = cls_out_channels
        self.group_reg_dims = group_reg_dims
        self.num_pred = num_pred
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            # cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        # if self.normedlinear:
        #     cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        # else:
        #     # cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        #     cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = RegLayer(self.embed_dims, self.num_reg_fcs, self.group_reg_dims)

        self.cls_branches = nn.ModuleList(
            [copy.deepcopy(fc_cls) for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [copy.deepcopy(reg_branch) for _ in range(self.num_pred)])
        
        load_weights(self.cls_branches, cls_branches_state_dict_path)
        load_weights(self.reg_branches, reg_branches_state_dict_path)



