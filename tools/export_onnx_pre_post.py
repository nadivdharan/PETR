import argparse
import sys
from pathlib import Path
import torch
import os
import numpy as np
import onnx
from onnxsim import simplify
import torch.nn.functional as F

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector

from mmdet.models.utils.transformer import inverse_sigmoid


sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D
from projects.mmdet3d_plugin.models.dense_heads.petrv2_head import pos2posemb3d


from typing import List, Dict

class MeanTimeStamp(torch.nn.Module):
    def __init__(self, batch_size:int=1, num_cams:int=6):
        super().__init__()
        self.batch_size = batch_size
        self.num_cams = num_cams

        self.dummy_proj = torch.zeros(batch_size, num_cams, 256, 10, 25)

    @torch.jit.script
    # def forward(img_metas: List[torch.FloatTensor]):
    def forward(img_metas: List[float]):
        batch_size = 1
        num_cams = 6
    # def forward(self, img_metas: Dict[str, List[torch.tensor]]):
    # def forward(self, img_metas):
        time_stamps = []
        # time_stamps.append(img_metas)
        time_stamps = torch.cat([torch.tensor(img_metas)])

        # for img_meta in img_metas:    
        #     # time_stamps.append(np.asarray(img_meta['timestamp']))
        #     # import ipdb; ipdb.set_trace()
        #     time_stamps.append(img_meta['timestamp'])
        # time_stamp = self.dummy_proj.new_tensor(time_stamps)
        time_stamp = torch.tensor(time_stamps)
        # time_stamp = torch.FloatTensor(time_stamps)
        time_stamp = time_stamp.view(batch_size, -1, num_cams)
        mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)

        return mean_time_stamp


class Petr3D_Split(Petr3D):
    def __init__(self, split=None, cfg_dict=None, batch_size=1, num_cams=6):
        super(Petr3D_Split, self).__init__(**cfg_dict)
        self.onnx_split_choices = ['backbone', 'transformer', 'postprocess', 'middle-process']
        assert split in self.onnx_split_choices, f"{split} is not one of {self.onnx_split_choices}"
        self.split = split

        self.batch_size = batch_size
        self.num_cams = num_cams

        # self.mean_time_stamp = None
        # if split == 'postprocess':
        #     self.mean_time_stamp = MeanTimeStamp(batch_size=batch_size, num_cams=num_cams)

    
    def forward(self, x, img_metas):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        if self.split == 'backbone':
            return self.foward_backbone(x, img_metas)
        elif self.split == 'transformer':
            return  self.foward_transformer(x, img_metas)
        elif self.split == 'postprocess':
            return  self.forward_postprocess(x, img_metas)
        elif self.split == 'middle-process':
            return  self.forward_middle_process(x, img_metas)
        else:
            raise ValueError(f"Unsupported split type {self.split}. Valid values are {self.onnx_split_choices}")

    def foward_backbone(self, img, img_metas):
        img = [img] if img is None else img
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        return img_feats

    def foward_transformer(self, img_feats, img_metas):
        outs = self.pts_bbox_head(img_feats, img_metas)
        return outs
    
    def forward_middle_process(self, mlvl_feats, img_metas):
        # import ipdb; ipd.set_trace()
        x = mlvl_feats[self.pts_bbox_head.position_level]
        batch_size, num_cams = x.size(0), x.size(1)

        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                # import ipdb; ipdb.set_trace()
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
            
        x = self.pts_bbox_head.input_proj(x.flatten(0,1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])
        # import ipdb; ipdb.set_trace()
        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)

        if self.pts_bbox_head.with_position:
            coords_position_embeding, _ = self.pts_bbox_head.position_embeding(mlvl_feats, img_metas, masks)
            if self.pts_bbox_head.with_fpe:
                coords_position_embeding = self.pts_bbox_head.fpe(coords_position_embeding.flatten(0,1), x.flatten(0,1)).view(x.size())

            pos_embed = coords_position_embeding

            if self.pts_bbox_head.with_multiview:
                sin_embed = self.pts_bbox_head.positional_encoding(masks)
                sin_embed = self.pts_bbox_head.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
            else:
                pos_embeds = []
                for i in range(num_cams):
                    xy_embed = self.pts_bbox_head.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(xy_embed.unsqueeze(1))
                sin_embed = torch.cat(pos_embeds, 1)
                sin_embed = self.pts_bbox_head.adapt_pos3d(sin_embed.flatten(0, 1)).view(x.size())
                pos_embed = pos_embed + sin_embed
        else:
            if self.pts_bbox_head.with_multiview:
                pos_embed = self.pts_bbox_head.positional_encoding(masks)
                pos_embed = self.pts_bbox_head.adapt_pos3d(pos_embed.flatten(0, 1)).view(x.size())
            else:
                pos_embeds = []
                for i in range(num_cams):
                    pos_embed = self.pts_bbox_head.positional_encoding(masks[:, i, :, :])
                    pos_embeds.append(pos_embed.unsqueeze(1))
                pos_embed = torch.cat(pos_embeds, 1)

        reference_points = self.pts_bbox_head.reference_points.weight
        query_embeds = self.pts_bbox_head.query_embedding(pos2posemb3d(reference_points))
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1) #.sigmoid()
        return x, pos_embed

    
    # def forward_postprocess(self, input_proj, outs_dec, img_metas):
    def forward_postprocess(self, outs_dec, img_metas):
        # out_dec --> torch.Size([num_dec_layers, batch_size, num_queries, embed_dim])
        reference_points = self.pts_bbox_head.reference_points.weight
        reference_points = reference_points.unsqueeze(0).repeat(self.batch_size, 1, 1) #.sigmoid()

        if self.pts_bbox_head.with_time:
            time_stamps = []
            for img_meta in img_metas:    
                time_stamps.append(np.asarray(img_meta['timestamp']))
            # time_stamp = input_proj.new_tensor(time_stamps)
            time_stamp = torch.Tensor(time_stamps)
            time_stamp = time_stamp.view(self.batch_size, -1, self.num_cams)
            mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)
            # assert self.mean_time_stamp is not None, f"mean_time_stamp is not initiazlied"
            # import ipdb; ipdb.set_trace()
            # mean_time_stamp = self.mean_time_stamp(img_metas)
            # mean_time_stamp = self.mean_time_stamp([float(x) for x in img_metas])

        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.pts_bbox_head.cls_branches[lvl](outs_dec[lvl])
            tmp = self.pts_bbox_head.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:2] += reference[..., 0:2]
            tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
            tmp[..., 4:5] += reference[..., 2:3]
            tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

            if self.pts_bbox_head.with_time:
                tmp[..., 8:] = tmp[..., 8:] / mean_time_stamp[:, None, None]

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)

        all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (self.pts_bbox_head.pc_range[3] - self.pts_bbox_head.pc_range[0]) + self.pts_bbox_head.pc_range[0])
        all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (self.pts_bbox_head.pc_range[4] - self.pts_bbox_head.pc_range[1]) + self.pts_bbox_head.pc_range[1])
        all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (self.pts_bbox_head.pc_range[5] - self.pts_bbox_head.pc_range[2]) + self.pts_bbox_head.pc_range[2])

        # return all_cls_scores, all_bbox_preds
        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'enc_cls_scores': None,
            'enc_bbox_preds': None, 
        }
        return outs




def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='model config path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('-o', '--opset', type=int, default=13)
    parser.add_argument('--ncams', type=int, default=6, help='Number of cameras')
    parser.add_argument('--img-dims', nargs=2, type=int, default=None, metavar=('height','width'), help='Change input resolution (height, width). Overrides cfg values')
    parser.add_argument('--mha-groups', type=int, default=4, help='How many groups to split the multi attention heads into')
    parser.add_argument('--split', default='transformer', type=str, help="Split and export backbone / transformer part of model",
                        choices=['backbone', 'transformer', 'postprocess', 'middle-process'], required=True)
    parser.add_argument('--out_name', default='petr_v2.onnx', type=str, help="Name for the onnx output")
    args = parser.parse_args()
    return args


def main(device='cpu'):
# def main(device='cuda:0'):
    args = parse_args()

    if args.ncams > 6:
        raise ValueError(f"Maximum number of cameras is 6 but got {args.ncam}")

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

    # sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                plg_lib = importlib.import_module(_module_path)

    # build the model and load checkpoint
    if args.img_dims is not None:
        print(f'Changing input resolution from'
              f' {cfg.ida_aug_conf.final_dim} to {tuple(args.img_dims)}')
        cfg.data.test.pipeline[2]['data_aug_conf']['final_dim'] = args.img_dims
    cfg.model.pts_bbox_head.transformer.decoder.\
        transformerlayers.attn_cfgs[1].num_head_split = args.mha_groups
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    # get example data from dataset
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)
    for i, data in enumerate(data_loader):
        img_metas = data['img_metas'][0].data
        img = data['img'][0].data[0].to(device)
        B, N, C, H, W = img.shape
        if args.split == 'transformer' or args.split == 'middle-process':
            out_stride = 16
            C = cfg.model.pts_bbox_head.in_channels
            N = 2 * args.ncams
            img = [torch.zeros(B, N, C, H//out_stride, W//out_stride),
                   torch.zeros(B, N, C, H//(2*out_stride), W//(2*out_stride))]
    
            if img[0].shape[1] < 12:
                for k, v in img_metas[0][0].items():
                    if isinstance(v, list):
                        img_metas[0][0][k] = v[:img[0].shape[1]]
        elif args.split == 'postprocess':
            num_queries = cfg.model.pts_bbox_head.num_query
            embed_dims = cfg.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[1].embed_dims
            num_dec_layers = cfg.model.pts_bbox_head.transformer.decoder.num_layers
            # torch.Size([num_dec_layers, batch_size, num_queries, embed_dim])
            img = torch.zeros(num_dec_layers, B, num_queries, embed_dims)
        # elif args.split == 'middle_postprocess':
        #     img = [torch.zeros(B, N, C, H//out_stride, W//out_stride),
        #            torch.zeros(B, N, C, H//(2*out_stride), W//(2*out_stride))]
            
        break


    # convert img metas to torch tensors
    keys_to_remove = []
    for i in range(len(img_metas)):
        for j in range(len(img_metas[i])):
            for key, value in img_metas[i][j].items():
                if isinstance(value, np.ndarray):
                    img_metas[i][j][key] = torch.from_numpy(value)
                elif isinstance(value, list):
                    for k in range(len(value)):
                        if isinstance(value[k], np.ndarray):
                            img_metas[i][j][key][k] = torch.from_numpy(value[k])
                        elif isinstance(value, tuple):
                            img_metas[i][j][key][k] = torch.tensor(value[k])
                elif isinstance(value, dict):
                    for kk, vv in value.items():
                        if isinstance(vv, np.ndarray):
                            img_metas[i][j][key][kk] = torch.from_numpy(vv)
                elif isinstance(value, type):
                    keys_to_remove.append(key)
    for remove in keys_to_remove:
        img_metas[0][0].pop(remove)
    img_metas[0][0]['box_mode_3d'] = int(img_metas[0][0]['box_mode_3d'])
    
    # split backbone / transformer part of model
    from projects.mmdet3d_plugin.models.backbones.repvgg import RepVGG, repvgg_model_convert
    cfg.model.pop('type')
    split_model = Petr3D_Split(args.split, cfg.model)
    split_model.load_state_dict(model.state_dict())    
    if isinstance(split_model.img_backbone, RepVGG):
        img_backbone_deploy = repvgg_model_convert(model.img_backbone)
        split_model.img_backbone = img_backbone_deploy
        split_model.img_backbone.deploy = True
    split_model.eval()
    # split_model.to(device)

    # export ONNX
    with torch.no_grad():
        torch.onnx.export(split_model,
                          (img, img_metas[0]), args.out_name,
                        #   (img, img_metas[0][0]['timestamp']), args.out_name,
                          do_constant_folding=False,
                          opset_version=args.opset)
        model_onnx = onnx.load(args.out_name)
        model_simp, check = simplify(model_onnx)
        onnx.save(model_simp, args.out_name)
        print(f"Simplified model saved at: {args.out_name}")


if __name__ == '__main__':
    main()
