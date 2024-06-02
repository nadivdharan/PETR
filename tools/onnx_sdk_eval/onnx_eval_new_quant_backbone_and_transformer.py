import torch
import torch.nn.functional as F
from torch import nn
import os
from pathlib import Path
import argparse
import numpy as np
import json
from time import sleep, time

import onnxruntime

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet.datasets import replace_ImageToTensor
from mmcv import Config
from mmcv.runner import init_dist

from onnx_eval_utils import Coords3DPE, QueryEmbed, pos2posemb3d, load_weights, inverse_sigmoid, PredictionLayer


# MODEL_BACKBONE_PATH = '/workspace/PETR/onnx/petrv2_vovnet_800x320_backbone_tmp.onnx'
# MODEL_TRANSFORMER_PATH = '/workspace/PETR/onnx/petrv2_vovnet_800x320_transformer_tmp.onnx'


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) an ONNX model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0', 'cpu'], help='GPU / CPU')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--dist', action='store_true')
    parser.add_argument('--out', help='output result file in pickle format', required=True, default=None)
    # parser.add_argument('--preprocess-in', help='ONNX including transformer pre-processing')
    parser.add_argument('--pre-post-weights', default=None, help='pre/post processing modules weights paths')
    parser.add_argument('--backbone-path', default=None, help='Backbone Path')
    parser.add_argument('--transformer-path', default=None, help='Transformer Path')
    parser.add_argument('--backbone-results', default=None, help='Backbone inference results dir')
    parser.add_argument('--transformer-results', default=None, help='Transformer inference results dir')

    # parser.add_argument('checkpoint', help='checkpoint file')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def load_cfg(cfg_path, samples_per_gpu=1):
    import sys
    sys.path.insert(0, '/workspace/PETR/')
    cfg = Config.fromfile(cfg_path)
    # if args.cfg_options is not None:
        # cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    
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
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(cfg_path)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)
    return cfg


def run_backbone(sess, img_pp):
    input_name = sess.get_inputs()[0].name
    output_names = [x.name for x in sess.get_outputs()]
    img_feats = sess.run(output_names, {input_name: img_pp.cpu().numpy()})
    return img_feats

def build_transformer_inputs(input_names, img_feats, img_metas, pos_embed=None, bn=False):
    inputs = {}
    # import ipdb; ipdb.set_trace()
    if pos_embed is not None:
        if bn:
            inputs[input_names[0]] = np.expand_dims(pos_embed.detach().cpu().numpy(), axis=0).transpose(0,3,2,1)
            inputs[input_names[1]] = np.expand_dims(img_feats.detach().cpu().numpy(), axis=0).transpose(0,3,2,1)
        else:
            bs, _, c, _, _ = img_feats.shape
            inputs[input_names[0]] = np.expand_dims(pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, c).detach().cpu().numpy(), axis=0).transpose(0,3,2,1)
            inputs[input_names[1]] = np.expand_dims(img_feats.permute(1, 3, 4, 0, 2).reshape(-1, bs, c).detach().cpu().numpy(), axis=0).transpose(0,3,2,1)
        # inputs[input_names[0]] = pos_embed.reshape(1, 256, 1, -1).detach().cpu().numpy()
        # inputs[input_names[1]] = img_feats.reshape(1, 256, 1, -1).detach().cpu().numpy()
    else:
        inputs[input_names[0]] = img_feats.cpu().numpy()
        for i, shape in enumerate(img_metas[0]['img_shape']):
            inputs[input_names[2*i+1]] = np.array(shape[0])
            inputs[input_names[2*i+2]] = np.array(shape[1])
        inputs[input_names[2*i+3]], inputs[input_names[2*i+4]], _ = [np.array(x) for x in img_metas[0]['pad_shape'][0]]

    return inputs

def transformer_preprocess(img_feats, img_metas, cfg, position_level, device='cuda:0', state_dict_paths=None):
    #state_dict_paths = json.load(open("petrv2-preprocess_state_dicts.json"))

    batch_size = img_feats[0].size(0)
    embed_dims = 256
    input_norm = False
    if hasattr(cfg.model.pts_bbox_head.transformer,"input_norm"):
        input_norm = True
    preprocess_transformer = Coords3DPE(in_channels=cfg.model.pts_bbox_head.in_channels,
                                        position_level=position_level,
                                        embed_dims=embed_dims,
                                        input_norm=input_norm,
                                        state_dict_paths=state_dict_paths['coords_3dpe']).to(device).eval()
    with torch.no_grad():
        pos_embed, img_feats_projection = preprocess_transformer(img_metas, img_feats)

    num_query = cfg.model.pts_bbox_head.num_query
    # query_embedding = QueryEmbed(embed_dims=embed_dims,
    #                              num_query=num_query,
    #                              state_dict_path=state_dict_paths['query_embed']).to(device)

    # reference_embed = nn.Embedding(query_embedding.num_query, 3).to(device)
    reference_embed = nn.Embedding(num_query, 3).to(device)
    load_weights(reference_embed, state_dict_paths['refernce_embed'])
    reference_points = reference_embed.weight   
    # query_embeds = query_embedding(pos2posemb3d(reference_points))
    reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1) #.sigmoid()

    # print("Preprcoess doesn't break...")
    return pos_embed, img_feats_projection, reference_points


def transformer_postprocess(outs_dec, x, img_metas, reference_points=None, state_dict_paths=None, post_norm_sd=None, input_norm=False, shape=None):
    # import ipdb; ipdb.set_trace()
    if input_norm:
        x = x.reshape(*shape)#.permute(3,0,4,1,2)
    batch_size = x.size(0)
    for i, out_dec in enumerate(outs_dec):
        out_dec = out_dec.squeeze(0).transpose(1, 2, 0)
        outs_dec[i] = out_dec
    time_stamps = []
    for img_meta in img_metas:    
        time_stamps.append(np.asarray(img_meta['timestamp']))
    time_stamp = x.new_tensor(time_stamps).to(reference_points.device)
    time_stamp = time_stamp.view(batch_size, -1, 6)
    # time_stamp = time_stamp.view(batch_size, -1, 4)
    mean_time_stamp = (time_stamp[:, 1, :] - time_stamp[:, 0, :]).mean(-1)

    outputs_classes = []
    outputs_coords = []
    # TODO merge with preprocess:
    cls_branches_state_dict_path = state_dict_paths["cls_branches"]
    reg_branches_state_dict_path = state_dict_paths["reg_branches"]

    # TODO take from cfg
    embed_dims = 256
    num_classes = 10
    detect = PredictionLayer(num_classes,
                             embed_dims=embed_dims,
                             cls_branches_state_dict_path=cls_branches_state_dict_path,
                             reg_branches_state_dict_path=reg_branches_state_dict_path).to(x.device).eval()

    outs_dec = torch.from_numpy(np.array(outs_dec)).to(x.device)
    if post_norm_sd is not None:
        outs_dec = torch.mul(outs_dec, post_norm_sd['decoder.post_norm.weight']) + post_norm_sd['decoder.post_norm.bias']
    for lvl in range(outs_dec.shape[0]):
        reference = inverse_sigmoid(reference_points.clone())
        assert reference.shape[-1] == 3
        with torch.no_grad():
            outputs_class = detect.cls_branches[lvl](outs_dec[lvl])
            tmp = detect.reg_branches[lvl](outs_dec[lvl])

        tmp[..., 0:2] += reference[..., 0:2]
        tmp[..., 0:2] = tmp[..., 0:2].sigmoid()
        tmp[..., 4:5] += reference[..., 2:3]
        tmp[..., 4:5] = tmp[..., 4:5].sigmoid()

        tmp[..., 8:] = tmp[..., 8:] / mean_time_stamp[:, None, None]

        outputs_coord = tmp
        outputs_classes.append(outputs_class)
        outputs_coords.append(outputs_coord)
    
    all_cls_scores = torch.stack(outputs_classes)
    all_bbox_preds = torch.stack(outputs_coords)

    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
    all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2])

    outs = {
        'all_cls_scores': all_cls_scores,
        'all_bbox_preds': all_bbox_preds,
        'enc_cls_scores': None,
        'enc_bbox_preds': None, 
    }
    return outs

def run_transformer(sess, img_feats, img_metas,
                    pos_embed=None, position_level=0, device='cuda:0',
                    reference_points=None,
                    bn=False):
    input_names = [x.name for x in sess.get_inputs()]
    output_names = [x.name for x in sess.get_outputs()]

    inputs = build_transformer_inputs(input_names, img_feats, img_metas,
                                      pos_embed=pos_embed, bn=bn)
    # print("Running transformer session")
    outs_dec = sess.run(output_names, inputs)

    return outs_dec
    

def eval_one(idx, data, sess_backbone, sess_transformer, cfg=None, device='cuda:0', pre_post_weights_paths_dict=None, infer_results=None, model_name=None):
    img_in = data['img'][0].data[0].to(device)
    img_metas = data['img_metas'][0].data[0]

    position_level = 0
    if hasattr(cfg.model.pts_bbox_head, 'position_level'):
        assert cfg.model.pts_bbox_head.position_level in [0, 1]
        position_level = cfg.model.pts_bbox_head.position_level

    # import ipdb; ipdb.set_trace()
    if sess_backbone is not None:
        # print("Running bacbone ONNX")
        img_feats = run_backbone(sess_backbone, img_in)
    else:
        # Load backbone inference results
        # img_feats = []
        # for i in range(2):
        #     res_path = os.path.join(infer_results['backbone'], f"{model_name}_img_feats{i}_{idx}.npy")
        #     # assert Path(res_path).is_file(), f"Could not find file {res_path}"
        #     if not Path(res_path).is_file():
        #         # Assuming single output - creating dummy 2nd output
        #         assert i==1, "Something went wrong with backbone outputs..."
        #         # img_feat = np.array([0])
        #         img_feat = img_feat
        #     else:
        #         img_feat = np.load(res_path, mmap_mode="r")
        #     img_feats.append(img_feat)
        res_path = os.path.join(infer_results['backbone'], f"{model_name}_img_feats0_{idx}.npy")
        while not Path(res_path).is_file():
            print(f'Waiting for {res_path} to be created...', end='\r')
            sleep(0.75)
        print(f"\n{res_path} CREATED !")
        success = False
        while not success:
            try:
                img_feat = np.load(res_path)
                success = True
            except:
                pass
        # position_level = cfg.model.pts_bbox_head.position_level
        img_feats = [np.array([0]), img_feat] if position_level==1 else [img_feat, np.array([0])]
        

    for i, img_feat in enumerate(img_feats):
        img_feat = torch.tensor(img_feat).to(device)
        img_feats[i] = img_feat

    img_feats_proj = img_feats
    shape = None
    pos_embed = None
    input_norm = False
    if pre_post_weights_paths_dict is not None:
        state_dict_paths = pre_post_weights_paths_dict['preprocess']
        if "batch_norms" in state_dict_paths['coords_3dpe']:
            input_norm = True
            assert cfg.model.pts_bbox_head.transformer.input_norm
            shape = np.array(img_feats_proj[position_level].shape)
            embed_dims = cfg.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[0].embed_dims
            assert cfg.model.pts_bbox_head.transformer.decoder.transformerlayers.attn_cfgs[1].embed_dims == embed_dims
            shape[2] = embed_dims
        pos_embed, img_feats_proj, reference_points = transformer_preprocess(img_feats, img_metas, cfg, position_level,
                                                                    device=device, state_dict_paths=state_dict_paths)

    # if sess_backbone is not None and sess_transformer is None:
    if sess_backbone is not None:
    # if False:
        #  assert model_name is not None
         # Save backbone results to feed quantized transformer model in separate env
         inputs = {}
         if input_norm:
            inputs['input_layer1'] = pos_embed.detach().cpu().numpy().transpose(1,0,2)
            inputs['input_layer2'] = img_feats_proj.detach().cpu().numpy().transpose(1,0,2)
         else:
            bs, _, ch, _, _ = img_feats_proj.shape
            inputs['input_layer1'] = pos_embed.permute(1, 3, 4, 0, 2).reshape(-1, bs, ch).detach().cpu().numpy().transpose(1,0,2)
            inputs['input_layer2'] = img_feats_proj.permute(1, 3, 4, 0, 2).reshape(-1, bs, ch).detach().cpu().numpy().transpose(1,0,2)
         t0 = time()
         np.savez(os.path.join(infer_results['backbone'], f'{model_name}_transformer_input_{idx}.npz'), **inputs)
         t1 = time()
         print(f'Saving took {(t1-t0):.3f} seconds')


    if sess_transformer is not None:
    # if sess_transformer is None:
        # print("Running transformer ONNX")
        outs_dec = run_transformer(sess_transformer,
                            img_feats_proj, img_metas,
                            pos_embed=pos_embed,
                            device=device,
                            reference_points=reference_points,
                            bn=input_norm
                            )
        # import ipdb; ipdb.set_trace()
    else:
        print("\nLoading results for transformer")
        inputs_path = os.path.join(infer_results['transformer'], f"{model_name}_transformer_output_{idx}.npy")
        while not os.path.exists(inputs_path):
            print(f'Waiting for {inputs_path} to be created...', end='\r')
            sleep(0.75)
        print(f"\n{inputs_path} CREATED !")
        success = False
        while not success:
            try:
                tmp = np.load(inputs_path, mmap_mode="r")
                success = True
            except:
                pass
        outs_dec = []
        for out_dec in tmp:
            outs_dec.append(out_dec)
    # TODO Add post norm cfg to state dict oaths json
    post_norm_sd = None
    if ('post_norm_cfg' not in cfg.model.pts_bbox_head.transformer.decoder or
        cfg.model.pts_bbox_head.transformer.decoder['post_norm_cfg'] is not None):                
        # state_dict_path = "/home/nadivd/workspace/PETR/pre_post/petrv2_vovnet_800x320_post_norm_state_dict.npz"
        state_dict_path = pre_post_weights_paths_dict['postprocess']['post_norm']
        post_norm_sd = np.load(state_dict_path, allow_pickle=True)['arr_0'][()]

    if pos_embed is not None:
        # outs = transformer_postprocess(outs_dec, inputs[input_names[1]], reference_points=reference_points)
        state_dict_paths = pre_post_weights_paths_dict['postprocess']
        outs = transformer_postprocess(outs_dec, img_feats_proj, img_metas,
                                       reference_points=reference_points,
                                       state_dict_paths=state_dict_paths,
                                       post_norm_sd=post_norm_sd,
                                       input_norm=input_norm,
                                       shape=shape)

    else:
        bbox_pts = outs_dec
        all_cls_scores, all_bbox_preds = bbox_pts
        outs = {
                'all_cls_scores': torch.tensor(all_cls_scores),
                'all_bbox_preds': torch.tensor(all_bbox_preds),
                'enc_cls_scores': None,
                'enc_bbox_preds': None,
    }
    # print("Transformer done...")

    # outs = np.load('/home/nadivd/workspace/PETR/pp_data/debug/outs_test.npz', allow_pickle=True)['arr_0'][()]
    from projects.mmdet3d_plugin.core.bbox.coders.nms_free_coder import NMSFreeCoder
    # TODO from CFG
    point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    voxel_size = [0.2, 0.2, 8]
    nms_free_coder = NMSFreeCoder(post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
                                  pc_range=point_cloud_range,
                                  max_num=300,
                                  voxel_size=voxel_size,
                                  num_classes=10)
    rescale = True  # TODO from CFG
    bbox_list = get_bboxes(outs, img_metas, nms_free_coder, rescale=rescale)
    bbox_results = [
        bbox3d2result(bboxes, scores, labels)
        for bboxes, scores, labels in bbox_list
    ]

    bbox_list = [dict() for i in range(len(img_metas))]
    for result_dict, pts_bbox in zip(bbox_list, bbox_results):
        result_dict['pts_bbox'] = pts_bbox
    
    return bbox_list


def main(args, samples_per_gpu=1):
    if (args.out is None) or (not args.out.endswith(".npy")):
        raise ValueError("Please use --out and provide .npy path to save eval results")
    distributed = args.dist
    device = args.device
    cfg = load_cfg(args.config, samples_per_gpu=samples_per_gpu)
    print("Building dataset...")
    from time import time
    t0 = time()
    dataset = build_dataset(cfg.data.test)
    t1 = time()
    print(f"Dataset Build took {t1-t0} sec")

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed,
        shuffle=False)
  
    if distributed:
        init_dist('pytorch', **cfg.dist_params)
    
    results = []
    dataset = data_loader.dataset
    import mmcv
    from mmcv.runner import get_dist_info
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    sleep(2)  # This line can prevent deadlock problem in some cases.

    pre_post_weights_paths_dict = None
    if args.pre_post_weights is not None:
        with open(args.pre_post_weights) as f:
            pre_post_weights_paths_dict = json.load(f)
    
    infer_results = {'backbone': args.backbone_results,
                     'transformer': args.transformer_results}
    providers = ['CUDAExecutionProvider']

    model_name = None
    sess_backbone, sess_transformer = None, None
    backbone_onnx_path = args.backbone_path
    transformer_onnx_path = args.transformer_path
    if args.backbone_path.endswith('.onnx'):
        sess_backbone = onnxruntime.InferenceSession(backbone_onnx_path, providers=providers)
    if args.transformer_path.endswith('.onnx'):
        sess_transformer = onnxruntime.InferenceSession(transformer_onnx_path, providers=providers)

    bb_suf = Path(backbone_onnx_path).suffix
    assert bb_suf in ['.onnx', '.har'], f"Unrecognized file type {backbone_onnx_path}"
    t_suf = Path(transformer_onnx_path).suffix
    assert t_suf in ['.onnx', '.har'], f"Unrecognized file type {transformer_onnx_path}"
    if bb_suf == '.har':
        model_name = Path(backbone_onnx_path).name.split(bb_suf)[0]
    elif t_suf == '.har':
        model_name = Path(transformer_onnx_path).name.split(t_suf)[0]
    if (model_name is not None) and (model_name.endswith('_optimized')):
        model_name = model_name.split('_optimized')[0]
    print()
    print(f"Model Name: {model_name}")
    print()
    for i, data in enumerate(data_loader):
        result = eval_one(i, data, sess_backbone, sess_transformer, cfg, device, pre_post_weights_paths_dict, infer_results, model_name)
        results.extend(result)
        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()
        # if i>10:
            # break
    
    print(f"\n Inference done for {i+1} images... Running evaluation... ")
    np.save(args.out, np.array(results)) # e.g. repvgg_a1_results_bb_native_petr_onnx.npy
    tmp_res = np.load(args.out, allow_pickle=True)
    for res in tmp_res:
        res['pts_bbox']['boxes_3d'].tensor.requires_grad = False
    from mmdet.apis.test import collect_results_cpu
    if distributed:
        tmp_results = collect_results_cpu(tmp_res, len(dataset), tmpdir=None)
    if rank == 0:
        evaluate(tmp_results, dataset, cfg)
    print(f"\n Evaluation done for {i+1} images... ")

    if rank==0:
        import ipdb; ipdb.set_trace()
    return

    for res in results:
        res['pts_bbox']['boxes_3d'].tensor.requires_grad = False

    from mmdet.apis.test import collect_results_cpu
    if distributed:
        results = collect_results_cpu(results, len(dataset), tmpdir=None)
    if rank == 0:
        if args.out:
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        evaluate(results, dataset, cfg)

def evaluate(outputs, dataset, cfg, **kwargs):
    eval_kwargs = cfg.get('evaluation', {}).copy()
    # hard-code way to remove EvalHook args
    for key in [
            'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
            'rule'
    ]:
        eval_kwargs.pop(key, None)
    eval_kwargs.update(dict(metric='bbox', **kwargs))
    print(dataset.evaluate(outputs, **eval_kwargs))


def get_bboxes(preds_dicts, img_metas, bbox_coder, rescale=False):
    """Generate bboxes from bbox head predictions.
    Args:
        preds_dicts (tuple[list[dict]]): Prediction results.
        img_metas (list[dict]): Point cloud and image's meta info.
    Returns:
        list[dict]: Decoded bbox, scores and labels after nms.
    """
    preds_dicts = bbox_coder.decode(preds_dicts)
    num_samples = len(preds_dicts)

    ret_list = []
    for i in range(num_samples):
        preds = preds_dicts[i]
        bboxes = preds['bboxes']
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
        scores = preds['scores']
        labels = preds['labels']
        ret_list.append([bboxes, scores, labels])
    return ret_list

def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_3d=bboxes.to('cpu'),
        scores_3d=scores.cpu(),
        labels_3d=labels.cpu())

    if attrs is not None:
        result_dict['attrs_3d'] = attrs.cpu()

    return result_dict

if __name__ == '__main__':
    args = parse_args()
    main(args)


'''
time CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=29501 tools/onnx_eval_new.py projects/configs/petrv2/petrv2_vovnet_gridmask_p4_800x320.py
'''


'''
time CUDA_VISIBLE_DEVICES=2 python3 -m torch.distributed.launch --nproc_per_node 1 --master_port=29502 ~/workspace/PETR/onnx/onnx_eval/onnx_eval_new_quant.py projects/configs/petrv2/petrv2_repvgg_a1.py --backbone-path ~/workspace/PETR/compile/backbones/repvgg_a1_800x320_12_images/ft2/petrv2_repvgg_a1_320x800_backbone_optimized.har --transformer-path ~/workspace/PETR/onnx/petrv2_repvgg_a1/repvgg_a1_transformer_const0.onnx --pre-post-weights ~/workspace/PETR/pre_post/repvgg_a1/petrv2_pre_post_state_dicts.json --dist --backbone-results ~/workspace/PETR/onnx/onnx_eval/results/repvgg_a1_320x800/optimized/ft2/again/ --out repvgg_a1_results_bb_optimized_ft2_again_petr_onnx.npy
'''