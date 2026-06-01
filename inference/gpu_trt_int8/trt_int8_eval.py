"""TensorRT INT8 Engine 构建 + 评测（MinMax 校准，和 BPU max 对齐）"""
import os, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import tensorrt as trt
import numpy as np, cv2, time, glob
import pycuda.driver as cuda
import pycuda.autoinit
from pathlib import Path

PROJECT = Path(r"F:\RDKX5投弹")
ONNX = PROJECT / "deploy/yolov8n_2cls_nchw.onnx"
ENGINE = PROJECT / "deploy/yolov8n_2cls_nchw_int8.trt"
CAL_DIR = PROJECT / "deploy/cal_f32"
VAL_IMG = PROJECT / "val/images"
VAL_LAB = PROJECT / "val/labels"
BATCH = 4; NC = 2; REG_MAX = 16; INP = 640
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# ============================================================
# 1. 校准器（MinMax，和 BPU max 一致）
# ============================================================
class MinMaxCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, files, batch_size, cache_file):
        super().__init__()
        self.files = files
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current = 0
        self.device_input = cuda.mem_alloc(batch_size * 3 * 640 * 640 * 4)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current >= len(self.files):
            return None
        end = min(self.current + self.batch_size, len(self.files))
        batch = []
        for i in range(self.current, end):
            data = np.fromfile(str(self.files[i]), dtype=np.float32)
            data = data.reshape(3, 640, 640) / 255.0
            batch.append(data)
        self.current = end
        if len(batch) < self.batch_size:
            # pad with last image
            while len(batch) < self.batch_size:
                batch.append(batch[-1].copy())
        batch_np = np.stack(batch).astype(np.float32).ravel()
        cuda.memcpy_htod(self.device_input, batch_np)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        if Path(self.cache_file).exists():
            return Path(self.cache_file).read_bytes()
        return None

    def write_calibration_cache(self, cache):
        Path(self.cache_file).write_bytes(cache)

# ============================================================
# 2. 构建 INT8 Engine
# ============================================================
def build_engine():
    print("=== 构建 TensorRT INT8 Engine ===")
    print(f"ONNX: {ONNX}")
    print(f"校准图片: {len(list(CAL_DIR.glob('*.f32')))} 张")

    if ENGINE.exists():
        print(f"Engine 已存在 ({ENGINE.stat().st_size/1024:.0f} KB)，跳过构建")
        return

    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(ONNX, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX 解析错误: {parser.get_error(i)}")
            raise RuntimeError("ONNX 解析失败")

    # 打印输入输出
    print(f"  输入: {network.get_input(0).name} {network.get_input(0).shape}")
    for i in range(network.num_outputs):
        print(f"  输出[{i}]: {network.get_output(i).name} {network.get_output(i).shape}")

    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.INT8)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)

    cal_files = sorted(list(CAL_DIR.glob("*.f32")))[:50]
    cache_file = str(PROJECT / "deploy/calibration.cache")
    calibrator = MinMaxCalibrator(cal_files, BATCH, cache_file)
    config.int8_calibrator = calibrator

    print("  构建中...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("构建失败")
    ENGINE.write_bytes(serialized)
    print(f"  Engine 已保存: {ENGINE.name} ({ENGINE.stat().st_size/1024:.0f} KB)")

# ============================================================
# 3. INT8 推理 + 评测
# ============================================================
def load_engine():
    runtime = trt.Runtime(TRT_LOGGER)
    with open(ENGINE, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    return engine

def infer_one(engine, img, ow, oh):
    """单张推理，返回预测框列表"""
    context = engine.create_execution_context()

    # 预处理：letterbox RGB NCHW 0-1
    tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = cv2.resize(tensor, (640, 640))
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)).reshape(1, 3, 640, 640)

    # 分配输入输出 GPU 内存
    input_shape = engine.get_tensor_shape(engine.get_tensor_name(0))
    d_input = cuda.mem_alloc(int(np.prod(input_shape) * 4))
    d_outputs = []
    h_outputs = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        if name == engine.get_tensor_name(0):
            continue
        shape = engine.get_tensor_shape(name)
        size = int(np.prod(shape))
        d_out = cuda.mem_alloc(size * 4)
        h_out = np.zeros(size, dtype=np.float32)
        d_outputs.append((name, d_out))
        h_outputs.append((name, shape, h_out))

    # 拷入
    cuda.memcpy_htod(d_input, tensor.ravel())

    # 设置地址
    context.set_tensor_address(engine.get_tensor_name(0), int(d_input))
    for name, d_out in d_outputs:
        context.set_tensor_address(name, int(d_out))

    # 推理
    stream = cuda.Stream()
    context.execute_async_v3(stream.handle)
    stream.synchronize()

    # 拷出
    raw = []
    for name, shape, h_out in h_outputs:
        _, d_out = d_outputs[h_outputs.index((name, shape, h_out))]
        cuda.memcpy_dtoh(h_out, d_out)
        raw.append(h_out.reshape(shape))

    # 后处理（和板端一致）
    strides = [8, 16, 32]
    ab, as0, ac = [], [], []
    for si, s in enumerate(strides):
        cls_buf = raw[si * 2 + 1]
        bbox_buf = raw[si * 2]
        H, W = cls_buf.shape[2], cls_buf.shape[3]
        cls = cls_buf.transpose(0, 2, 3, 1).reshape(-1, NC)
        scores = 1 / (1 + np.exp(-cls))
        mx = scores.max(1)
        mc = scores.argmax(1)
        valid = np.flatnonzero(mx > 0.25)
        if len(valid) == 0:
            continue
        v_scores = mx[valid]
        v_cls = mc[valid]
        box = bbox_buf.transpose(0, 2, 3, 1).reshape(-1, REG_MAX * 4)
        b = box[valid]
        bmax = b.reshape(-1, 4, REG_MAX).max(axis=2, keepdims=True)
        e = np.exp(b.reshape(-1, 4, REG_MAX) - bmax)
        dfl = (e / e.sum(axis=2, keepdims=True) * np.arange(REG_MAX)).sum(axis=2)
        gy, gx = np.unravel_index(valid, (H, W))
        grid = np.stack([gx + 0.5, gy + 0.5], axis=1).astype(np.float32)
        x1y1 = (grid - dfl[:, :2]) * s
        x2y2 = (grid + dfl[:, 2:]) * s
        boxes = np.concatenate([x1y1, x2y2], axis=1)
        boxes[:, [0, 2]] *= ow / INP
        boxes[:, [1, 3]] *= oh / INP
        ab.append(boxes)
        as0.append(v_scores)
        ac.append(v_cls)

    if not ab:
        return np.array([]), np.array([]), np.array([])

    boxes = np.concatenate(ab)
    scores = np.concatenate(as0)
    cls_ids = np.concatenate(ac)
    bb = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
    fb, fs, fc = [], [], []
    for cid in np.unique(cls_ids):
        if cid >= NC:
            continue
        idx = cls_ids == cid
        ind = cv2.dnn.NMSBoxes(bb[idx].tolist(), scores[idx].tolist(), 0.25, 0.45)
        if len(ind) > 0:
            ind = np.array(ind).flatten()
            fb.append(boxes[idx][ind])
            fs.append(scores[idx][ind])
            fc.append(cls_ids[idx][ind])
    if not fb:
        return np.array([]), np.array([]), np.array([])
    return np.concatenate(fb), np.concatenate(fs), np.concatenate(fc)


def evaluate():
    print("\n=== TensorRT INT8 评测 ===")

    def load_gt(lp, iw, ih):
        bs, cs = [], []
        if not os.path.exists(lp):
            return bs, cs
        for line in open(lp).read().strip().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            bs.append([int((cx - w / 2) * iw), int((cy - h / 2) * ih), int((cx + w / 2) * iw), int((cy + h / 2) * ih)])
            cs.append(c)
        return bs, cs

    def iou(b1, b2):
        x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        return inter / ((b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter + 1e-6)

    def match(pb, ps, pc, gb, gc):
        gm = [False] * len(gb)
        dm = [False] * len(pb)
        for i in np.argsort(ps)[::-1]:
            bi, bj = 0, -1
            for j, (b, c) in enumerate(zip(gb, gc)):
                if gm[j] or c != pc[i]:
                    continue
                v = iou(pb[i], b)
                if v > bi:
                    bi, bj = v, j
            if bi >= 0.5:
                dm[i] = True
                gm[bj] = True
        tp = sum(dm)
        fp = len(pb) - tp
        fn = len(gb) - sum(gm)
        return tp, fp, fn

    engine = load_engine()
    all_ims = sorted(list(VAL_IMG.glob("*.jpg")))[:50]
    tp = fp = fn = 0
    t0 = time.time()

    for ip in all_ims:
        bn = ip.stem
        lp = VAL_LAB / (bn + ".txt")
        img = cv2.imdecode(np.fromfile(str(ip), dtype=np.uint8), cv2.IMREAD_COLOR)
        oh, ow = img.shape[:2]
        gb, gc = load_gt(lp, ow, oh)

        pb, ps, pc = infer_one(engine, img, ow, oh)
        t, f, fn0 = match(pb, ps, pc, gb, gc)
        tp += t
        fp += f
        fn += fn0

    dt = time.time() - t0
    n = len(all_ims)
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    F1 = 2 * p * r / (p + r + 1e-6)
    print(f"  TensorRT INT8: {dt / n * 1000:.0f}ms/img  F1={F1:.4f}  P={p:.4f}  R={r:.4f}  TP={tp}  FP={fp}  FN={fn}")
    return F1


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    build_engine()
    evaluate()
