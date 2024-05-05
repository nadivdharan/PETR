import argparse
import torch
import torch.nn.functional as F
import numpy as np
import sys
import os
from pathlib import Path
from tqdm import tqdm

from mmcv import Config
from mmcv.runner import load_checkpoint, wrap_fp16_model
from mmcv.runner.base_module import BaseModule
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector


# sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))
sys.path.insert(0, os.getcwd())
from projects.mmdet3d_plugin.models.detectors.petr3d import Petr3D
from projects.mmdet3d_plugin.models.dense_heads.petrv2_head import pos2posemb3d


class DecoderInputs(BaseModule):
    def __init__(self, calib_set_size=1024):
        super(DecoderInputs, self).__init__()
        self.calib_set_size = calib_set_size
    
    def forward(self, x, mask, query_embed, pos_embed, reg_branch=None):
        """Forward function for `Transformer`.
        Args:
            x (Tensor): Input query with shape [bs, c, h, w] where
                c = embed_dims.
            mask (Tensor): The key_padding_mask used for encoder and decoder,
                with shape [bs, h, w].
            query_embed (Tensor): The query embedding for decoder, with shape
                [num_query, c].
            pos_embed (Tensor): The positional encoding for encoder and
                decoder, with the same shape as `x`.
        Returns:
        """
        bs, n, c, h, w = x.shape
        memory = x.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        pos_embed = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c) # [bs, n, c, h, w] -> [n*h*w, bs, c]
        query_embed = query_embed.unsqueeze(1).repeat(
            1, bs, 1)  # [num_query, dim] -> [num_query, bs, dim]
        mask = mask.view(bs, -1)  # [bs, n, h, w] -> [bs, n*h*w]
        target = torch.zeros_like(query_embed)

        tgt_len, bsz, embed_dim = target.shape
        assert bsz == bs
        src_len = memory.size(0)
        tmp_mlvl_feats = memory.permute(1, 2, 0).reshape(1, embed_dim, -1, src_len).permute(0, 2, 3, 1)
        tmp_pos_embed = pos_embed.permute(1, 2, 0).reshape(1, embed_dim, -1, src_len).permute(0, 2, 3, 1)
        tmp_mlvl_feats = tmp_mlvl_feats.cpu().detach().numpy()
        tmp_pos_embed = tmp_pos_embed.cpu().detach().numpy()

        return tmp_pos_embed, tmp_mlvl_feats


class CalibGenerator(Petr3D):
    def __init__(self, cfg_dict=None, calib_set_size=1024, device='cuda:0'):
        super(CalibGenerator, self).__init__(**cfg_dict)
        self.calib_set_size = calib_set_size
        self.device = device

    def forward(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        return self.simple_test(img_metas[0], img[0], **kwargs)
    
    def simple_test(self, img_metas, img=None, **kwargs):
        """Test function without augmentaiton."""
        img = img.to(self.device)
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        x = img_feats[self.pts_bbox_head.position_level]
        batch_size, num_cams = x.size(0), x.size(1)

        input_img_h, input_img_w, _ = img_metas[0]['pad_shape'][0]
        masks = x.new_ones(
            (batch_size, num_cams, input_img_h, input_img_w))
        for img_id in range(batch_size):
            for cam_id in range(num_cams):
                img_h, img_w, _ = img_metas[img_id]['img_shape'][cam_id]
                masks[img_id, cam_id, :img_h, :img_w] = 0
            
        x = self.pts_bbox_head.input_proj(x.flatten(0,1))
        x = x.view(batch_size, num_cams, *x.shape[-3:])

        # interpolate masks to have the same spatial shape with x
        masks = F.interpolate(
            masks, size=x.shape[-2:]).to(torch.bool)

        if self.pts_bbox_head.with_position:
            coords_position_embeding, _ = self.pts_bbox_head.position_embeding(img_feats, img_metas, masks)
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
        query_embeds = self.pts_bbox_head.query_embedding(pos2posemb3d(reference_points, num_pos_feats=self.pts_bbox_head.num_pos_feats))
        reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1) #.sigmoid()
        pos_embed, mlvl_feats = self.pts_bbox_head.transformer(x, masks, query_embeds, pos_embed, self.pts_bbox_head.reg_branches)

        return pos_embed, mlvl_feats


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='model config path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--calib-set-size', type=int, default=1024, help='calibration set size')
    parser.add_argument('--save-dir', type=str, default=None, help='Folder to save calibration sets')
    parser.add_argument('--net-name', type=str, default=None, help='Model name')
    args = parser.parse_args()
    return args


def main(device='cuda:0'):
    args = parse_args()
    calib_set_size = args.calib_set_size
    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True

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

    

    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    model.eval()

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    from projects.mmdet3d_plugin.models.backbones.repvgg import RepVGG, repvgg_model_convert
    cfg.model.pop('type')
    calib_model = CalibGenerator(cfg.model, device=device)
    calib_model.load_state_dict(model.state_dict())    
    if isinstance(calib_model.img_backbone, RepVGG):
        img_backbone_deploy = repvgg_model_convert(model.img_backbone)
        calib_model.img_backbone = img_backbone_deploy
        calib_model.img_backbone.deploy = True
    calib_model.pts_bbox_head.transformer = DecoderInputs()
    calib_model.eval()
    calib_model.to(device)

    calib_set = {'backbone': [],
                 'transformer':
                    {'pos_embed': [],
                     'mlvl_feats_proj': []
                    }
                }
    for i, data in tqdm(enumerate(data_loader), total=calib_set_size):
        if i + 1 == calib_set_size:
            break
        img_metas = data['img_metas'][0].data[0]
        img = data['img'][0].data[0].to(device)

        img_npy = img[0,0].cpu().numpy().transpose(1,2,0)
        shape = img_npy.shape
        calib_data = img_npy.reshape(1, *shape)

        assert cfg.model.pts_bbox_head.position_level in [0, 1]
        bb_stride = 16 if cfg.model.pts_bbox_head.position_level==0 else 32
        tokens = int((shape[0]/bb_stride) * (shape[1]/bb_stride) * (6*2))
        embed_dims =cfg.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[0].embed_dims

        if i==0:
            calib_set['backbone'] = np.zeros((calib_set_size, *shape))
            calib_set['transformer']['pos_embed'] = np.zeros((calib_set_size, 1, tokens, embed_dims))
            calib_set['transformer']['mlvl_feats'] = np.zeros((calib_set_size, 1, tokens, embed_dims))

        pos_embed, mlvl_feats = calib_model(img_metas=data['img_metas'][0].data, img=data['img'][0].data)

        calib_set['backbone'][i] = calib_data
        calib_set['transformer']['pos_embed'][i] = pos_embed
        calib_set['transformer']['mlvl_feats'][i] = mlvl_feats
    
    print('\nFinished')

    work_dir = os.getcwd() if args.save_dir is None else args.save_dir
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    suf = Path(args.config).suffix
    net_name = Path(args.config).name.split(suf)[0]if args.net_name is None else args.net_name

    # Save backbone calibration set
    calib_backbone_path = os.path.join(work_dir, f'{net_name}_calib_set_backbone_{calib_set_size}.npy')
    np.save(calib_backbone_path, calib_set['backbone'])

    # Save transformer calibration set
    calib_set_transformer = {
        net_name + '/input_layer1': calib_set['transformer']['pos_embed'],
        net_name + '/input_layer2': calib_set['transformer']['mlvl_feats']
    }
    calib_transformer_path = os.path.join(work_dir, f'{net_name}_calib_set_transformer_{calib_set_size}.npz')
    np.savez(calib_transformer_path, **calib_set_transformer)

    print(f"Calibration sets saved at:\nBackbone: {calib_backbone_path}\nTransformer: {calib_transformer_path}\n")
    return


if __name__ == "__main__":
    main()

"""
This script needs to be run from the PETR/ folder, e.g.
/workspace/PETR$ CUDA_VISIBLE_DEVICES=4 python3 /data/data/nadivd/PETR/calib/gen_calib_set.py projects/configs/petrv2/petrv2_fcos3d_repvgg_h2x32_decoder_3_UN_800x320.py ~/workspace/PETR/training/fcos3d_repvgg_h2/petrv2_fcos3d_repvgg_h2x32_decoder_3_UN_800x320/latest.pth
                   --calib-set-size 1024
                   --save-dir /data/data/nadivd/PETR/calib/fcos3d_repvgg_b0/
                   --net-name petrv2_repvgg_b0_transformer_x32_decoder_3_UN_800x320_split_1_const0
"""