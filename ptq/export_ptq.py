"""YOLOv8n 2cls NCHW PTQ: final cls+bbox conv layers on CPU (FP32)"""
import os, sys, subprocess, shutil
from pathlib import Path

DEPLOY = Path("/mnt/f/RDKX5投弹/deploy")
ONNX = DEPLOY / "yolov8n_2cls_nchw.onnx"
PREFIX = "yolov8n_2cls_nchw_cpuconv_bayese_640x640_nv12"

ws = DEPLOY / "hb_ws_cpuconv"
shutil.rmtree(ws, ignore_errors=True); ws.mkdir(parents=True)

# Only put FINAL conv layers on CPU (no SiLU after them)
# These are the direct model outputs - no downstream ops depend on them
cpu_nodes = """node_info: {
    "/model.22/cv3.0/cv3.0.2/Conv": {'ON': 'CPU'},
    "/model.22/cv2.0/cv2.0.2/Conv": {'ON': 'CPU'},
    "/model.22/cv3.1/cv3.1.2/Conv": {'ON': 'CPU'},
    "/model.22/cv2.1/cv2.1.2/Conv": {'ON': 'CPU'},
    "/model.22/cv3.2/cv3.2.2/Conv": {'ON': 'CPU'},
    "/model.22/cv2.2/cv2.2.2/Conv": {'ON': 'CPU'}
  }"""

cfg = f"""model_parameters:
  onnx_model: '{ONNX}'
  march: "bayes-e"
  {cpu_nodes}
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
  debug: true
  optimize_level: 'O3'
"""
(ws / "config.yaml").write_text(cfg)

print("=== hb_mapper PTQ: final conv on CPU, rest on BPU ===")
r = subprocess.run(["hb_mapper", "makertbin", "--config", "config.yaml", "--model-type", "onnx"], cwd=str(ws))
if r.returncode != 0: print("FAILED!"); sys.exit(1)

bpu = ws / "bpu_model_output"
for f in bpu.glob("*.bin"):
    dst = DEPLOY / f.name
    shutil.copy(f, dst)
    print(f"Output: {dst} ({dst.stat().st_size/1024:.0f} KB)")
print("Done!")
