"""WSL: 批量扫描分类偏置值"""
import onnx, numpy as np, subprocess, shutil
from pathlib import Path

DEPLOY = Path("/mnt/f/RDKX5~1/deploy")
ONNX_SRC = DEPLOY / "yolov8n_2cls_nchw.onnx"

# 测试偏置值范围
BIAS_VALUES = [-5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0]

for BIAS_VAL in BIAS_VALUES:
    tag = f"bias_{str(BIAS_VAL).replace('.', 'p')}"
    PREFIX = f"yolov8n_2cls_{tag}_bayese_640x640_nv12"
    ONNX_MOD = DEPLOY / f"yolov8n_2cls_nchw_{tag}.onnx"
    BIN_FILE = DEPLOY / f"{PREFIX}.bin"

    if BIN_FILE.exists():
        print(f"\n[跳过] {tag} - bin 已存在")
        continue

    print(f"\n=== 偏置={BIAS_VAL} ===")

    # 修改 ONNX
    m = onnx.load(str(ONNX_SRC))
    for node in m.graph.node:
        if node.op_type != "Conv": continue
        if "cv3" not in node.name or ".2/" not in node.name: continue
        w_init = [i for i in m.graph.initializer if i.name == node.input[1]]
        if not w_init or w_init[0].dims[0] != 2: continue
        bias_init = [i for i in m.graph.initializer if i.name == node.input[2]]
        if bias_init:
            new_bias = np.array([BIAS_VAL, BIAS_VAL], dtype=np.float32)
            bias_init[0].raw_data = new_bias.tobytes()
    onnx.save(m, str(ONNX_MOD))

    # 编译
    ws = DEPLOY / f"hb_ws_{tag}"
    shutil.rmtree(ws, ignore_errors=True)
    ws.mkdir(parents=True)

    cfg = f"""model_parameters:
  onnx_model: '{ONNX_MOD}'
  march: "bayes-e"
  working_dir: 'bpu_model_output'
  output_model_file_prefix: '{PREFIX}'
input_parameters:
  input_name: ""
  input_type_rt: 'nv12'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  norm_type: 'data_scale'
  scale_value: 0.003921568627451
calibration_parameters:
  cal_data_dir: '{DEPLOY / "cal_f32"}'
  cal_data_type: 'float32'
  calibration_type: 'max'
compiler_parameters:
  jobs: 16
  compile_mode: 'latency'
  debug: false
  optimize_level: 'O3'
"""
    (ws / "config.yaml").write_text(cfg)

    r = subprocess.run(["hb_mapper", "makertbin", "--config", "config.yaml", "--model-type", "onnx"],
                       cwd=str(ws), capture_output=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[-200:]}")
        continue

    bpu = ws / "bpu_model_output"
    for f in bpu.glob("*.bin"):
        shutil.copy(f, BIN_FILE)
        print(f"  => {BIN_FILE.name} ({BIN_FILE.stat().st_size/1024:.0f} KB)")

print("\n全部完成!")
