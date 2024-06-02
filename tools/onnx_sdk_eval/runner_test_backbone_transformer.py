import numpy as np
from tqdm import tqdm
import os
import argparse
import torch
from time import time, sleep

from hailo_sdk_client import ClientRunner, InferenceContext

import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--har', help='Input HAR path')
    parser.add_argument('--split-type', help='backbone / transformer', choices=['backbone', 'transformer'])
    parser.add_argument('--data', help='Data directory')
    parser.add_argument('--save', help='Save results directory')
    parser.add_argument('--target', help='Inference target', choices=['native', 'optimized'])
    parser.add_argument('--debug', help='Debug mode', action='store_true')
    args = parser.parse_args()
    return args


class Dataset(torch.utils.data.Dataset):
    def __init__(self, datadir=None, device='cuda:0', dataset_type='backbone', model_name=None):
        self.device = device
        self.datadir = datadir
        self.dataset_type = dataset_type
        self.model_name = model_name
        # '/data/data/nadivd/PETR/pp_np_imgs'

    def __len__(self):
        if self.dataset_type == 'backbone':
            names = [x for x in os.listdir(self.datadir) if x.endswith(".npy") and x.startswith("img") and 'meta' not in x]
        elif self.dataset_type == 'transformer':
            names = [x for x in os.listdir(self.datadir) if x.endswith('.npz') and x.startswith('transformer_input')]
        else:
            raise ValueError(f'Unsupported dataset type {self.dataset_type}')
        print(f"Total: {len(names)}")
        # return len(names)
        return  6019

    def __getitem__(self, idx):
        if self.dataset_type == 'backbone':
            img_path = f'{self.datadir}/img_{idx}.npy'
            assert os.path.exists(img_path), "could not find {img_path}"
            img = np.load(img_path, mmap_mode="r")
            return img
        elif self.dataset_type == 'transformer':
            assert self.model_name is not None
            inputs_path = os.path.join(self.datadir, f"{self.model_name}_transformer_input_{idx}.npz")
            while not os.path.exists(inputs_path):
                print(f'Waiting for {inputs_path} to be created...', end='\r')
                sleep(0.75)
            print(f"\n{inputs_path} CREATED !")
            # sleep(1.0)
            success = False
            while not success:
                try:
                    inputs = np.load(inputs_path, mmap_mode="r")
                    success = True
                    print(f"\n{inputs_path} LOADED successfully !")
                except:
                    pass
            return inputs
        else:
            raise ValueError(f'Unsupported dataset type {self.dataset_type}')
        
#'1.11.0+cu102''
def main(args):
    torch.backends.cudnn.benchmark = True
    har_path = args.har
    dataset_type = args.split_type
    assert os.path.isfile(har_path), f"HAR file {har_path} does not exists"
    #    img_path = args.img
    #    assert os.path.isfile(img_path), f"Image {img_path} does not exists"
    #    import ipdb; ipdb.set_trace()
    #    img_in = np.load(img_path).transpose(0,2,3,1)
    target = args.target
    if target=='native':
        infer_ctx = InferenceContext.SDK_NATIVE
    elif target=='optimized':
        infer_ctx = InferenceContext.SDK_QUANTIZED
    else:
        raise ValueError(f"target {target} not supported here")

    print("Prepering inference...")
    runner = ClientRunner(har=har_path)

    device = 'cuda:0'
    dataset = Dataset(args.data, device=device, dataset_type=dataset_type, model_name=runner.model_name)
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=1,
                                              shuffle=False,)
                                            #   num_workers=4)
    
    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        # if idx<3600:
            # continue
        with runner.infer_context(infer_ctx) as ctx:
            # sleep(1.0)
            # t0 = time()
            if args.split_type == 'backbone':
                img = data[0].permute(0,2,3,1).cpu().numpy()
                # img = data[0].transpose((0,2,3,1))
                result = runner.infer(ctx, img, batch_size=1)
                if isinstance(result, list) and len(result) == 2:
                    raise ValueError("You're using an out of date backbone :/")
                    result = [np.expand_dims(res, axis=0).transpose(0,1,4,2,3) for res in result]
                    np.save(os.path.join(args.save, runner.model_name + f'_img_feats0_{idx}.npy'), result[0])
                    np.save(os.path.join(args.save, runner.model_name + f'_img_feats1_{idx}.npy'), result[1])
                elif isinstance(result, np.ndarray):
                    result = np.expand_dims(result, axis=0).transpose(0,1,4,2,3)
                    np.save(os.path.join(args.save, runner.model_name + f'_img_feats0_{idx}.npy'), result)
                else:
                    raise ValueError(f"Expected result to be a list of size 2 or ndarray but got {type(result)}")
            elif args.split_type == 'transformer':
                # tmp = data['input_layer1']
                # data['input_layer1'] = data['input_layer2']
                # data['input_layer2'] = tmp
                result = runner.infer(ctx, data, batch_size=1)
                if args.debug:
                    hailo_model = runner.get_keras_model(ctx)._model
                    hailo_model.enable_internal_encoding()
                    @tf.function
                    def get_intermediate_results(data):
                        _ = hailo_model(data)
                        return hailo_model.interlayer_tensors
                    new_data = {f"{hailo_model.model_name}/{k}": v for k, v in data.items()}
                    int_res = get_intermediate_results(new_data)
                    hailo_model._debug_mode = True
                    _ = hailo_model(new_data)
                    int_res = hailo_model.internal_layer_outputs
                    
                    import ipdb; ipdb.set_trace()

                result = [res.transpose(0,3,1,2) for res in result]
                # import ipdb; ipdb.set_trace()
                t0 = time()
                np.save(os.path.join(args.save, f"{runner.model_name}_transformer_output_{idx}.npy"), np.array(result))
                t1 = time()
                print(f'Saving took {(t1-t0):.3f} seconds')
            else:
                raise ValueError(f'Unsupported split type {args.split_type}')


    print("Infer finished...")
    import ipdb; ipdb.set_trace()
    print()


if __name__ == '__main__':
    args = parse_args()
    main(args)

