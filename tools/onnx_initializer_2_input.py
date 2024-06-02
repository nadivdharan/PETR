In [1]: import onnx

In [2]: onnx_model = onnx.load("/home/nadivd/workspace/PETR/ForRonit/petrv2_repvgg_b0_x32_decoder_3_UN_800x320/petrv2_repvgg_b0_x32_decoder_3_UN_800x320_postprocess.onnx")


In [5]: [(x, i) for (i, x) in enumerate(onnx_model.graph.initializer) if "892" in x.name]
Out[5]: 
[(dims: 1
  dims: 1
  dims: 1
  data_type: 11
  name: "892"
  raw_data: "\354Q\270\036\205\353\363?",
  85)]

In [6]: div = onnx_model.graph.initializer.pop(85)

In [7]: div
Out[7]: 
dims: 1
dims: 1
dims: 1
data_type: 11
name: "892"
raw_data: "\354Q\270\036\205\353\363?"

In [8]: onnx_model.graph.input
Out[8]: 
[name: "x.1"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 3
      }
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 900
      }
      dim {
        dim_value: 256
      }
    }
  }
}
]

In [9]: a =  onnx_model.graph.input.pop(0)

In [12]: onnx_model.graph.input.insert(0, a)

In [13]: onnx_model.graph.input
Out[13]: 
[name: "x.1"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 3
      }
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 900
      }
      dim {
        dim_value: 256
      }
    }
  }
}
]

In [15]: a.name = "892"


In [20]: a.type.tensor_type.shape.dim.pop(0)
Out[20]: dim_value: 3

In [23]: sh = a.type.tensor_type.shape.dim.pop(0)

In [25]: sh = 1

In [26]: a.type.tensor_type.shape.dim.pop(0)
Out[26]: dim_value: 900

In [27]: a.type.tensor_type.shape.dim.pop(0)
Out[27]: dim_value: 256


In [32]: a.type.tensor_type.shape.dim
Out[32]: []


In [39]: onnx_model.graph.input
Out[39]: 
[name: "x.1"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 3
      }
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 900
      }
      dim {
        dim_value: 256
      }
    }
  }
}
]

In [40]: a = onnx_model.graph.input.pop(0)

In [41]: onnx_model.graph.input.insert(0, a)

In [42]: sh = a.type.tensor_type.shape.dim.pop(0)

In [43]: sh = a.type.tensor_type.shape.dim.pop(0)

In [44]: sh = a.type.tensor_type.shape.dim.pop(0)

In [45]: sh
Out[45]: dim_value: 900

In [46]: sh.dim_value = 1


In [48]: a.type.tensor_type.shape.dim.pop(0)
Out[48]: dim_value: 256


In [50]: a.type.tensor_type.shape.dim.insert(0, sh)

In [51]: a.type.tensor_type.shape.dim.insert(0, sh)

In [52]: a.type.tensor_type.shape.dim.insert(0, sh)

In [53]: a
Out[53]: 
name: "x.1"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 1
      }
    }
  }
}

In [54]: onnx_model.graph.input.insert(1, a)

In [55]: onnx_model.graph.input
Out[55]: 
[name: "x.1"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 3
      }
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 900
      }
      dim {
        dim_value: 256
      }
    }
  }
}
, name: "x.1"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 1
      }
    }
  }
}
]

In [56]: onnx_model.graph.input[1]
Out[56]: 
name: "x.1"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 1
      }
    }
  }
}

In [57]: onnx_model.graph.input[1].name
Out[57]: 'x.1'

In [58]: onnx_model.graph.input[1].name = "892"

In [59]: onnx_model.graph.input
Out[59]: 
[name: "x.1"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 3
      }
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 900
      }
      dim {
        dim_value: 256
      }
    }
  }
}
, name: "892"
type {
  tensor_type {
    elem_type: 1
    shape {
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 1
      }
      dim {
        dim_value: 1
      }
    }
  }
}
]

In [60]: onnx.save("modified.onnx", onnx_model)