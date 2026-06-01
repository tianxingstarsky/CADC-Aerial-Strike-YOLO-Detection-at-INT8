"""ONNX 分类头偏置修正：只改最终 cv3.*.2 Conv，设偏置为 +0.5"""
import onnx, numpy as np
from onnx import helper
from pathlib import Path

ONNX_SRC = Path(r"F:\RDKX5投弹\deploy\yolov8n_2cls_nchw.onnx")
ONNX_OUT = Path(r"F:\RDKX5投弹\deploy\yolov8n_2cls_nchw_posbias.onnx")
BIAS_VAL = 0.5

m = onnx.load(str(ONNX_SRC))

patched = 0
for node in m.graph.node:
    # 只改最终分类 Conv：op=Conv，包含 cv3.*.2/，weight 第一维 = 2
    if node.op_type != "Conv":
        continue
    if "cv3" not in node.name or ".2/" not in node.name:
        continue
    w_init = [i for i in m.graph.initializer if i.name == node.input[1]]
    if not w_init or w_init[0].dims[0] != 2:
        continue  # 跳过中间层 cv3.2.0, cv3.2.1

    has_bias = len(node.input) > 2
    bias_name = node.name + "_posbias"
    new_bias = np.array([BIAS_VAL, BIAS_VAL], dtype=np.float32)

    if has_bias:
        old_bias_name = node.input[2]
        old_init = [i for i in m.graph.initializer if i.name == old_bias_name]
        if old_init:
            old_val = np.frombuffer(old_init[0].raw_data, dtype=np.float32)
            old_init[0].raw_data = new_bias.tobytes()
            print(f"  {node.name}: bias {old_val} → {new_bias}")
            patched += 1
        else:
            print(f"  {node.name}: 偏置非 initializer，跳过")
    else:
        bias_init = helper.make_tensor(bias_name, onnx.TensorProto.FLOAT, [2], new_bias.tolist())
        m.graph.initializer.append(bias_init)
        node.input.append(bias_name)
        print(f"  {node.name}: 添加 bias → {new_bias}")
        patched += 1

if patched == 0:
    print("未找到目标 Conv！")
else:
    onnx.save(m, str(ONNX_OUT))
    print(f"\n已保存: {ONNX_OUT} ({ONNX_OUT.stat().st_size/1024:.0f} KB)")

    # 验证
    m2 = onnx.load(str(ONNX_OUT))
    for node in m2.graph.node:
        if node.op_type == "Conv" and "cv3" in node.name and ".2/" in node.name:
            w_init = [i for i in m2.graph.initializer if i.name == node.input[1]]
            if w_init and w_init[0].dims[0] == 2:
                bias_init = [i for i in m2.graph.initializer if i.name == node.input[2]]
                if bias_init:
                    b = np.frombuffer(bias_init[0].raw_data, dtype=np.float32)
                    print(f"  验证 {node.name}: bias={b}")
