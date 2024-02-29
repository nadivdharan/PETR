# Copyright (c) OpenMMLab. All rights reserved.
import argparse
# import time
import torch
import os
from mmcv import Config
# from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model

# from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_detector
# from mmdet3d.core import LiDARInstance3DBoxes

import numpy as np
import onnx
from onnxsim import simplify

onnx_split_choices = ['backbone', 'transformer']


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet deploy a model to ONNX format')
    parser.add_argument('config', help='test config file path', default='./projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py')
    parser.add_argument('checkpoint', help='checkpoint file', default='./ckpts/epoch_24_petrv2.pth')
    parser.add_argument('--no_simplify', action='store_false')
    # parser.add_argument('--shape', nargs=2, type=int, default=[800, 1333])
    parser.add_argument('-o', '--opset', type=int, default=13)
    parser.add_argument('--split', default='transformer', type=str, help="Split backbone or transformer parts of model",
                        choices=onnx_split_choices)
    parser.add_argument('--out_name', default='petr_v2.onnx', type=str, help="Name for the onnx output")

    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    # import modules from plguin/xx, registry will be updated
    import ipdb; ipdb.set_trace()
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
                print(_module_path)
                # import ipdb; ipdb.set_trace()
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # # build the dataloader
    # # TODO: support multiple images per gpu (only minor changes are needed)
    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=cfg.data.workers_per_gpu,
    #     dist=False,
    #     shuffle=False)

    # build the model and load checkpoint
    # import ipdb; ipdb.set_trace()
    cfg.model.train_cfg = None
    model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    # load_checkpoint(model, args.checkpoint, map_location='cpu')
    # if args.fuse_conv_bn:
    #     model = fuse_module(model)

    # model = MMDataParallel(model, device_ids=[0])

    model.eval()

    # the first several iterations may be very slow so skip them
    # num_warmup = 5
    # pure_inf_time = 0

    # benchmark with several samples and take the average
    # for i, data in enumerate(data_loader):

    #     torch.cuda.synchronize()
    #     start_time = time.perf_counter()

    #     with torch.no_grad():
    #         model(return_loss=False, rescale=True, **data)

    #     torch.cuda.synchronize()
    #     elapsed = time.perf_counter() - start_time

    #     if i >= num_warmup:
    #         pure_inf_time += elapsed
    #         if (i + 1) % args.log_interval == 0:
    #             fps = (i + 1 - num_warmup) / pure_inf_time
    #             print(f'Done image [{i + 1:<3}/ {args.samples}], '
    #                   f'fps: {fps:.1f} img / s')

    #     if (i + 1) == args.samples:
    #         pure_inf_time += elapsed
    #         fps = (i + 1 - num_warmup) / pure_inf_time
    #         print(f'Overall fps: {fps:.1f} img / s')
    #         break

    # dummy_input = torch.randn(6, 3, 224, 224)
    if args.split == 'backbone':
        img = [torch.zeros(1, 1, 3, 320, 800, dtype=torch.float32).to(device)]  # B, N, C, H, W  == [1, 12, 3, 320, 800]
    elif args.split == 'transformer':
        img = [torch.zeros(1, 12, 256, 20, 50, dtype=torch.float32).to(device),
            torch.zeros(1, 12, 256, 10, 25, dtype=torch.float32).to(device)]
    else:
        raise ValueError(f"split arg {args.split} is not one of {onnx_split_choices}")
    img_metas = np.load('img_metas.npy', allow_pickle=True).tolist()

    # This part might be reduntant. might be that N=12 is a must to due previous frame
    if img[0].shape[1] > 12:
        raise ValueError("img_metas has only N = 12")
    elif img[0].shape[1] < 12:
        for k, v in img_metas[0][0].items():
            if isinstance(v, list):
                img_metas[0][0][k] = v[:img[0].shape[1]]
    # import ipdb; ipdb.set_trace()
    # Convert numpy arrays in img_metas to PyTorch tensors
    keys_to_remove = []
    for i in range(len(img_metas)):
        for j in range(len(img_metas[i])):
            for key, value in img_metas[i][j].items():
                if isinstance(value, np.ndarray):
                    print(f"{key=} is np array")
                    img_metas[i][j][key] = torch.from_numpy(value)
                elif isinstance(value, list):
                    for k in range(len(value)):
                        if isinstance(value[k], np.ndarray):
                            print(f"{key}[{k}] is np array")
                            # if key == 'pad_shape':
                            #     import ipdb; ipdb.set_trace()
                            img_metas[i][j][key][k] = torch.from_numpy(value[k])
                        elif isinstance(value, tuple):
                            # import ipdb; ipdb.set_trace()
                            print(f"pad shape is tuple {key}") 
                            img_metas[i][j][key][k] = torch.tensor(value[k])
                elif isinstance(value, dict):
                    for kk, vv in value.items():
                        if isinstance(vv, np.ndarray):
                            print(f"{key}[{kk}] is np array")
                            # if key == 'pad_shape':
                            #     import ipdb; ipdb.set_trace()
                            img_metas[i][j][key][kk] = torch.from_numpy(vv)
                
                elif isinstance(value, type):
                    print(f"Removing {key} from img_metas[{i}][{j}]")
                    keys_to_remove.append(key)
    for remove in keys_to_remove:
        img_metas[0][0].pop(remove)
    img_metas[0][0]['box_mode_3d'] = int(img_metas[0][0]['box_mode_3d'])
    print("******************")
    for k, v in img_metas[0][0].items():
        print(k, type(v))
    
    # import ipdb; ipdb.set_trace()
    with torch.no_grad():
        torch.onnx.export(model,
                        (img_metas, img), args.out_name,  # (img_metas, img)
                        #   input_names=['img'],
                        #   output_names=['output'],
                        # training=torch.onnx.TrainingMode.PRESERVE,
                        do_constant_folding=False,
                        opset_version=args.opset)  # args.opset
        # if also simplify
        if args.no_simplify:
            model_onnx = onnx.load(args.out_name)
            model_simp, check = simplify(model_onnx)
            onnx.save(model_simp, args.out_name)
            # runner.logger.info(f"Simplified model saved at: {args.out_name}")
            print(f"Simplified model saved at: {args.out_name}")
        else:
            print(f"Model saved at: {args.out_name}")
            # runner.logger.info(f"Model saved at: {args.out_name}")


if __name__ == '__main__':
    main()
