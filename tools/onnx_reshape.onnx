import json
import argparse
import onnx
import torch
from onnx.compose import merge_models


class InputModel(torch.nn.Module):

    def forward(self, x):
        x = x.flatten(2).permute(2, 0, 1)
        return x


class MaskModel(torch.nn.Module):
    def __init__(self, num_query, num_heads, num_tokens, num_splits):
        super(MaskModel, self).__init__()
        self.num_query = num_query
        self.num_heads = num_heads
        self.num_tokens = num_tokens
        self.num_splits = num_splits

    def forward(self, x):
        # x = x.flatten(2).permute(2, 0, 1)
        # x = torch.reshape(x, (900, 8, 12000)).permute(1, 0, 2)
        x = torch.zeros((self.num_query, self.num_heads//self.num_splits, self.num_tokens)).permute(1, 0, 2)
        return x


class OutputModel(torch.nn.Module):
    def __init__(self, num_query, embed_dim):
        super(OutputModel, self).__init__()
        self.num_query = num_query
        self.embed_dim = embed_dim

    def forward(self, x):
        x = x.permute(1, 2, 0)
        x = x.reshape((1, self.embed_dim, 1, self.num_query))
        return x


def add_reshape(args):
    with open(args.cfg, 'r') as f:
        cfg = json.load(f)
    EMBED_DIM = cfg['EMBED_DIM']
    NUM_TOKENS = cfg['NUM_TOKENS']
    NUM_QUERY = cfg['NUM_QUERY']
    NUM_HEADS = cfg['NUM_HEADS']
    NUM_SPLITS = cfg['NUM_SPLITS']
    INPUT_NAMES = cfg['INPUT_NAMES']
    MASK_NAMES = cfg['MASK_NAMES']
    OUTPUT_NAMES = cfg['OUTPUT_NAMES']

    onnx_model = onnx.load_model(args.onnx)
    extractor = onnx.utils.Extractor(onnx_model)
    onnx_model = extractor.extract_model(
        input_names=INPUT_NAMES + MASK_NAMES,
        output_names=OUTPUT_NAMES,
    )

    torch.onnx.export(InputModel(), torch.randn(1, EMBED_DIM, 1, NUM_TOKENS), 'input1.onnx', opset_version=13)
    torch.onnx.export(InputModel(), torch.randn(1, EMBED_DIM, 1, NUM_TOKENS), 'input2.onnx', opset_version=13)
    torch.onnx.export(MaskModel(NUM_QUERY, NUM_HEADS, NUM_TOKENS, NUM_SPLITS),
                      torch.randn(1, (NUM_HEADS//NUM_SPLITS)*NUM_TOKENS, 1, NUM_QUERY), 'input3.onnx', opset_version=13)

    inp1 = onnx.load_model('input1.onnx')
    inp2 = onnx.load_model('input2.onnx')
    inp3 = onnx.load_model('input3.onnx')

    inp1.ir_version = onnx_model.ir_version
    inp2.ir_version = onnx_model.ir_version
    inp3.ir_version = onnx_model.ir_version

    onnx_model = merge_models(inp1, onnx_model, io_map=[(inp1.graph.output[0].name, INPUT_NAMES[0])], prefix1="a",)
    onnx_model = merge_models(inp2, onnx_model, io_map=[(inp2.graph.output[0].name, INPUT_NAMES[1])], prefix1="b")
    prefix = "c"
    for i in range(len(MASK_NAMES)):
        onnx_model = merge_models(inp3, onnx_model, io_map=[(inp3.graph.output[0].name, MASK_NAMES[i])],  prefix1=prefix)
        prefix = chr(ord(prefix)+1)

    torch.onnx.export(OutputModel(NUM_QUERY, EMBED_DIM), torch.randn(NUM_QUERY, 1, EMBED_DIM), 'out1.onnx', opset_version=13)

    out1 = onnx.load_model('out1.onnx')
    out1.ir_version = onnx_model.ir_version

    for i, out in enumerate(onnx_model.graph.output):
        onnx_model = merge_models(onnx_model, out1, io_map=[(out.name, out1.graph.input[0].name)], prefix2=str(i))

    onnx.save_model(onnx_model, args.out)
    print(f"Model saved at {args.out}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--onnx', help='Input ONNX path')
    parser.add_argument('--out', help='Output ONNX path')
    parser.add_argument('--cfg', help='PETR parsing config path (json)')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    add_reshape(args)
