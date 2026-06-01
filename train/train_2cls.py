"""YOLOv8n native 2-class + NCHW export + hb_mapper PTQ"""
import os, sys, subprocess, shutil, types
import numpy as np, cv2
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

PROJECT = Path(r"F:\RDKX5投弹")
DEPLOY = PROJECT / "deploy"
YAML2 = PROJECT / "dataset_target_2cls.yaml"

# Ensure 2-class YAML
YAML2.write_text(f"""path: {PROJECT}
train: train/images
val: val/images

nc: 2
names:
  - Blue_Target
  - Red_Target
""")

# Step 1: Train
PT = PROJECT / "runs/yolov8n_2cls_100ep/weights/best.pt"
if not PT.exists():
    print("=== Train YOLOv8n 2-class, 100 epochs ===")
    model = YOLO("yolov8n.pt")
    model.train(
        data=str(YAML2), epochs=100, imgsz=640, batch=32, device=0, workers=2,
        cache=True, patience=0,  # NO early stop
        project=str(PROJECT / "runs"), name="yolov8n_2cls_100ep", exist_ok=True,
        lr0=0.01, lrf=0.01, cos_lr=True, warmup_epochs=3,
        close_mosaic=15, mosaic=1.0, plots=False,
    )
    model.save(str(PT))
else:
    print(f"Training already done: {PT}")

# Step 2: Export ONNX (NCHW, no permute, bbox first)
print("\n=== Export ONNX (NCHW) ===")
ONNX = DEPLOY / "yolov8n_2cls_nchw.onnx"
m = YOLO(str(PT if PT.exists() else PROJECT / "runs/yolov8n_2cls/weights/best.pt"))

def Df(self, x):
    r = []
    for i in range(self.nl):
        r.append(self.cv2[i](x[i]))  # bbox NCHW
        r.append(self.cv3[i](x[i]))  # cls NCHW
    return r

def patch(mod):
    for n, c in mod.named_children():
        if type(c) == Detect: c.forward = types.MethodType(Df, c)
        patch(c)
patch(m.model.model)

m.export(format='onnx', simplify=False, opset=11)
default = PT.with_suffix('.onnx')
if default.exists(): shutil.move(str(default), str(ONNX))
print(f"ONNX: {ONNX} ({ONNX.stat().st_size/1024:.0f} KB)")

# Step 3: Letterbox calibration data (.f32, 0-255)
cal_dir = DEPLOY / "cal_f32"
shutil.rmtree(cal_dir, ignore_errors=True); cal_dir.mkdir(parents=True)
for p in list((PROJECT / "train/images").glob("*.jpg"))[:100]:
    img = cv2.imread(str(p))
    if img is None: continue
    h, w = img.shape[:2]
    s = min(640/h, 640/w); nh, nw = int(h*s), int(w*s)
    res = cv2.resize(img, (nw, nh))
    canvas = np.full((640, 640, 3), 114, np.uint8)
    top, left = (640-nh)//2, (640-nw)//2
    canvas[top:top+nh, left:left+nw] = res
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    nchw = rgb.astype(np.float32).transpose(2, 0, 1)
    nchw.tofile(str(cal_dir / (p.stem + '.f32')))

print(f"Cal data: {cal_dir}")

# Step 4: hb_mapper PTQ (max calibration)
print("\n=== hb_mapper PTQ ===")
PREFIX = "yolov8n_2cls_nchw_bayese_640x640_nv12"
ws = DEPLOY / "hb_ws_2cls"
shutil.rmtree(ws, ignore_errors=True); ws.mkdir(parents=True)

cfg = f"""model_parameters:
  onnx_model: '{ONNX}'
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
  cal_data_dir: '{cal_dir}'
  cal_data_type: 'float32'
  calibration_type: 'max'
compiler_parameters:
  jobs: 16
  compile_mode: 'latency'
  debug: true
  optimize_level: 'O3'
"""
(ws / "config.yaml").write_text(cfg)

r = subprocess.run(["hb_mapper", "makertbin", "--config", "config.yaml", "--model-type", "onnx"], cwd=str(ws))
if r.returncode != 0: print("FAILED!"); sys.exit(1)

bpu = ws / "bpu_model_output"
for f in bpu.glob("*.bin"):
    shutil.copy(f, DEPLOY / f.name)
    print(f"Output: {DEPLOY / f.name} ({f.stat().st_size/1024:.0f} KB)")

print("Done!")
