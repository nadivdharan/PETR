import onnx

onnx_model = onnx.load('/home/nadivd/workspace/PETR/demo/ncams_6/x32_BN_q_304_dec_3_UN_800x320_split_8/transformer_adapt_pos3d_pp/test1/petrv2_b0_transformer_x32_BN_q_304_dec_3_UN_800x320_adapt_pos3d.onnx')
extractor = onnx.utils.Extractor(onnx_model)
onnx_model = extractor.extract_model(input_names=["3114","2913", "2741","3366"], output_names=["3410","3706","4002"])
onnx_model.graph.input
x = onnx_model.graph.input.pop(0)
onnx_model.graph.input.insert(0, x)
x
onnx_model = onnx.load('/home/nadivd/workspace/PETR/demo/ncams_6/x32_BN_q_304_dec_3_UN_800x320_split_8/transformer_adapt_pos3d_pp/test1/petrv2_b0_transformer_x32_BN_q_304_dec_3_UN_800x320_adapt_pos3d.onnx')
extractor = onnx.utils.Extractor(onnx_model)
onnx_model = extractor.extract_model(input_names=["3114","2913", "2741","3366"], output_names=["3410","3706","4002"])
onnx_model.graph.input
x = onnx_model.graph.input.pop(0)
onnx_model.graph.input.insert(0, x)
x
x.name="2913"
x.type.tensor_type.shape.dim
sh = x.type.tensor_type.shape.dim.pop(0)
sh = x.type.tensor_type.shape.dim.pop(0)
sh = x.type.tensor_type.shape.dim.pop(0)
sh = x.type.tensor_type.shape.dim.pop(0)
sh
sh.dim_value=300
x.type.tensor_type.shape.dim.insert(0,sh)
sh.dim_value=10
x.type.tensor_type.shape.dim.insert(0,sh)
sh.dim_value=384
x.type.tensor_type.shape.dim.insert(0,sh)
sh.dim_value=1
x.type.tensor_type.shape.dim.insert(0,sh)
x
onnx_model.graph.input.insert(1, x)
onnx_model.graph.input
onnx_model.graph.input.pop(2)
inp1 = onnx.load_model('input1.onnx')
inp1.graph.input
inp3 = onnx.load_model('input3.onnx')
inp3.graph.input
inp1.ir_version = onnx_model.ir_version
inp3.ir_version = onnx_model.ir_version
onnx_model = merge_models(inp1, onnx_model, io_map=[(inp1.graph.output[0].name, "3114")], prefix1="a")
from onnx.compose import merge_models
onnx_model = merge_models(inp1, onnx_model, io_map=[(inp1.graph.output[0].name, "3114")], prefix1="a")
onnx_model = merge_models(inp3, onnx_model, io_map=[(inp3.graph.output[0].name, "3366")],  prefix1="c")
onnx.checker.check_model(onnx_model)

onnx.save_model(onnx_model, '/home/nadivd/workspace/PETR/demo/ncams_6/x32_BN_q_304_dec_3_UN_800x320_split_8/transformer_adapt_pos3d_pp/test1/petrv2_b0_transformer_x32_BN_q_304_dec_3_UN_800x320_adapt_pos3d_const0.onnx')
