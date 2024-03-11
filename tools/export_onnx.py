import argparse
import sys
from pathlib import Path
import torch
import os
import numpy as np
import onnx
from onnxsim import simplify

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D


class Petr3D_Split(Petr3D):
    def __init__(self, split=None, cfg_dict=None):
        super(Petr3D_Split, self).__init__(**cfg_dict)
        self.onnx_split_choices = ['backbone', 'transformer']
        assert split in self.onnx_split_choices, f"{split} is not one of {self.onnx_split_choices}"
        self.split = split
    
    def forward(self, x, img_metas):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        if self.split == 'backbone':
            return self.foward_backbone(x, img_metas)
        elif self.split == 'transformer':
            return  self.foward_transformer(x, img_metas)
        else:
            raise ValueError(f"Unsupported split type {self.split}. Valid values are {self.onnx_split_choices}")

    def foward_backbone(self, img, img_metas):
        img = [img] if img is None else img
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        return img_feats

    def foward_transformer(self, img_feats, img_metas):
        outs = self.pts_bbox_head(img_feats, img_metas)
        return outs


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='model config path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('-o', '--opset', type=int, default=13)
    parser.add_argument('--split', default='transformer', type=str, help="Split and export backbone / transformer part of model",
                        choices=['backbone', 'transformer'])
    parser.add_argument('--out_name', default='petr_v2.onnx', type=str, help="Name for the onnx output")
    args = parser.parse_args()
    return args


def main(device='cpu'):
    args = parse_args()

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
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
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
        if args.split == 'transformer':
            B, N, C, H, W = img.shape
            out_stride = 16
            C = cfg.model.img_neck.out_channels
            img = [torch.zeros(B, N, C, H//out_stride, W//out_stride),
                   torch.zeros(B, N, C, H//(2*out_stride), W//(2*out_stride))]
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

    # export ONNX
    with torch.no_grad():
        torch.onnx.export(split_model,
                          (img, img_metas[0]), args.out_name,
                          do_constant_folding=False,
                          opset_version=args.opset)
        model_onnx = onnx.load(args.out_name)
        model_simp, check = simplify(model_onnx)
        onnx.save(model_simp, args.out_name)
        print(f"Simplified model saved at: {args.out_name}")


if __name__ == '__main__':
    main()
