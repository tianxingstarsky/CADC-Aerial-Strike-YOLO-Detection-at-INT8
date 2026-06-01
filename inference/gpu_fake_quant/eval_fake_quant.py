"""
固定 scale 真量化：校准 → 评估 → QAT 训练
用校准数据算好每个 Conv2d 的固定 scale，
推理和训练时不再逐样本动态算 scale。
"""
import os, sys, json, types, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
import numpy as np, cv2, time
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

PROJECT = Path(r"F:\RDKX5投弹")
DEPLOY = PROJECT / "deploy"
BEST_PT = PROJECT / "runs/yolov8n_2cls/weights/best.pt"
VAL_IMG = PROJECT / "val/images"
VAL_LAB = PROJECT / "val/labels"
CAL_DIR = DEPLOY / "cal_f32"
SCALES_FILE = DEPLOY / "fixed_scales.pt"
DEVICE = "cuda:0"
NC = 2; CONF = 0.25; NMS_IOU = 0.45; REG_MAX = 16; INP = 640


def load_model_and_patch(pt_path=BEST_PT):
    """加载模型并 patch Detect 头为 NCHW bbox-first 输出"""
    model = YOLO(str(pt_path))
    m = model.model
    m.eval()

    # Monkey-patch Detect forward for NCHW bbox-first
    detect = None
    for child in m.modules():
        if isinstance(child, Detect):
            detect = child
            break
    if detect is not None:

        def Df(self, x):
            r = []
            for i in range(self.nl):
                r.append(self.cv2[i](x[i]))
                r.append(self.cv3[i](x[i]))
            return r

        detect.forward = types.MethodType(Df, detect)
    return model, m


def calibrate():
    """校准阶段：用校准图片跑一遍，为每个 Conv2d 收集激活的全局 max"""
    print("=" * 60)
    print("阶段 1: 校准 - 采集每个 Conv2d 的固定 scale")

    model, m = load_model_and_patch()
    m.to(DEVICE).eval()

    # 记录每个 Conv2d 的激活 max
    act_max = {}

    def make_hook(name):
        def hook(module, inp, out):
            val = inp[0].detach().abs().max().item()
            if name not in act_max:
                act_max[name] = val
            else:
                act_max[name] = max(act_max[name], val)

        return hook

    hooks = []
    for mod_name, mod in m.named_modules():
        if isinstance(mod, nn.Conv2d):
            h = mod.register_forward_hook(make_hook(mod_name))
            hooks.append(h)

    # 加载校准图片
    cal_files = sorted(list(CAL_DIR.glob("*.f32")))[:50]
    print(f"  校准图片数: {len(cal_files)}")

    with torch.no_grad():
        for i, f in enumerate(cal_files):
            nchw = np.fromfile(str(f), dtype=np.float32).reshape(1, 3, 640, 640).copy()
            tensor = torch.from_numpy(nchw / 255.0).to(DEVICE)
            m(tensor)
            if (i + 1) % 10 == 0:
                print(f"  校准进度: {i + 1}/{len(cal_files)}")

    for h in hooks:
        h.remove()

    # 计算 scale = max / 127
    scales = {}
    for name, vmax in act_max.items():
        if vmax <= 0:
            vmax = 1e-6
        scales[name] = round(vmax / 127.0, 10)

    torch.save(scales, SCALES_FILE)
    print(f"  已保存 {len(scales)} 个固定 scale → {SCALES_FILE}")

    # 打印一些关键层的 scale
    head_names = [n for n in scales if "cv2" in n or "cv3" in n]
    print("\n  检测头 Conv scale:")
    for n in sorted(head_names):
        print(f"    {n}: scale={scales[n]:.6f}  max={act_max[n]:.4f}")

    return scales


def evaluate_fixed(scales=None):
    """用固定 scale 做假量化推理评估"""
    print("\n" + "=" * 60)
    print("阶段 2: 固定 scale 假量化推理评估")

    if scales is None:
        if SCALES_FILE.exists():
            scales = torch.load(SCALES_FILE, map_location="cpu")
        else:
            print("  无校准文件，请先运行 calibrate()")
            return

    model, m = load_model_and_patch()

    # 用固定 scale 替换每个 Conv2d 的 forward
    conv_list = list(m.named_modules())
    wrapped = 0
    for mod_name, mod in conv_list:
        if not isinstance(mod, nn.Conv2d):
            continue
        if mod_name not in scales:
            continue

        fixed_scale = scales[mod_name]

        def make_qf(conv=mod, s=fixed_scale):
            def qf(x):
                dev = conv.weight.device
                w = conv.weight
                w_abs = w.abs().reshape(w.size(0), -1).amax(dim=1).clamp(min=1e-6)
                z = torch.zeros(w.size(0), dtype=torch.int32, device=dev)
                w_q = torch.fake_quantize_per_channel_affine(
                    w,
                    scale=w_abs / 127.0,
                    zero_point=z,
                    axis=0,
                    quant_min=-127,
                    quant_max=127,
                )
                # 固定 scale 量化输入（和 BPU 一致）
                x_q = torch.fake_quantize_per_tensor_affine(
                    x,
                    scale=s,
                    zero_point=0,
                    quant_min=-128,
                    quant_max=127,
                )
                return nn.functional.conv2d(
                    x_q,
                    w_q,
                    conv.bias,
                    conv.stride,
                    conv.padding,
                    conv.dilation,
                    conv.groups,
                )

            return qf

        mod.forward = make_qf()
        wrapped += 1

    print(f"  已包装 {wrapped} 个 Conv2d（固定 scale）")

    m.to(DEVICE).eval()
    return run_eval(m, "固定 scale（校准max/127）")


def evaluate_dynamic():
    """用动态 scale（逐样本）做假量化推理评估（和之前一样）"""
    print("\n" + "=" * 60)
    print("阶段 2b: 动态 scale 假量化推理评估（对比基准）")

    model, m = load_model_and_patch()

    wrapped = 0
    for mod_name, mod in m.named_modules():
        if not isinstance(mod, nn.Conv2d):
            continue

        def make_qf(conv=mod):
            def qf(x):
                dev = conv.weight.device
                w = conv.weight
                w_abs = w.abs().reshape(w.size(0), -1).amax(dim=1).clamp(min=1e-6)
                z = torch.zeros(w.size(0), dtype=torch.int32, device=dev)
                w_q = torch.fake_quantize_per_channel_affine(
                    w,
                    scale=w_abs / 127.0,
                    zero_point=z,
                    axis=0,
                    quant_min=-127,
                    quant_max=127,
                )
                s = x.abs().amax().clamp(min=1e-6) / 127.0
                x_q = torch.fake_quantize_per_tensor_affine(
                    x, scale=s, zero_point=0, quant_min=-128, quant_max=127
                )
                return nn.functional.conv2d(
                    x_q,
                    w_q,
                    conv.bias,
                    conv.stride,
                    conv.padding,
                    conv.dilation,
                    conv.groups,
                )

            return qf

        mod.forward = make_qf()
        wrapped += 1

    print(f"  已包装 {wrapped} 个 Conv2d（动态 scale）")

    m.to(DEVICE).eval()
    return run_eval(m, "动态 scale（逐样本 amax/127）")


def run_eval(m, label):
    """跑评测，返回 (F1, P, R, TP, FP, FN)"""
    # 评测度量函数
    def load_gt(lp, iw, ih):
        bs, cs = [], []
        if not os.path.exists(lp):
            return bs, cs
        for line in open(lp).read().strip().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
            bs.append(
                [
                    int((cx - w / 2) * iw),
                    int((cy - h / 2) * ih),
                    int((cx + w / 2) * iw),
                    int((cy + h / 2) * ih),
                ]
            )
            cs.append(c)
        return bs, cs

    def iou(b1, b2):
        x1, y1, x2, y2 = (
            max(b1[0], b2[0]),
            max(b1[1], b2[1]),
            min(b1[2], b2[2]),
            min(b1[3], b2[3]),
        )
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        return inter / (
            (b1[2] - b1[0]) * (b1[3] - b1[1])
            + (b2[2] - b2[0]) * (b2[3] - b2[1])
            - inter
            + 1e-6
        )

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

    all_ims = sorted(list(VAL_IMG.glob("*.jpg")))[:50]
    tp = fp = fn = 0
    t0 = time.time()

    with torch.no_grad():
        for ip in all_ims:
            bn = ip.stem
            lp = VAL_LAB / (bn + ".txt")
            img = cv2.imread(str(ip))
            oh, ow = img.shape[:2]
            gb, gc = load_gt(lp, ow, oh)

            tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            tensor = cv2.resize(tensor, (640, 640))
            tensor = (
                torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
            )
            raw = m(tensor)

            if isinstance(raw, dict):
                raw = list(raw.values())
            if isinstance(raw, tuple):
                raw = list(raw)

            # NCHW 后处理
            strides = [8, 16, 32]
            ab, as0, ac = [], [], []
            for si, stride in enumerate(strides):
                cls_buf = raw[si * 2 + 1]
                bbox_buf = raw[si * 2]
                H, W = cls_buf.shape[2], cls_buf.shape[3]
                cls = cls_buf.permute(0, 2, 3, 1).reshape(-1, cls_buf.shape[1]).cpu().numpy()
                rc = cls[:, :NC]
                mx = rc.max(1)
                mc = rc.argmax(1)
                valid = np.flatnonzero(mx > -np.log(1.0 / 0.25 - 1.0))
                if len(valid) == 0:
                    continue
                scores = 1 / (1 + np.exp(-mx[valid]))
                vcls = mc[valid]

                box = (
                    bbox_buf.permute(0, 2, 3, 1)
                    .reshape(-1, REG_MAX * 4)
                    .cpu()
                    .numpy()
                )
                b = box[valid]
                bmax = b.reshape(-1, 4, REG_MAX).max(axis=2, keepdims=True)
                e = np.exp(b.reshape(-1, 4, REG_MAX) - bmax)
                dfl = (e / e.sum(axis=2, keepdims=True) * np.arange(REG_MAX)).sum(
                    axis=2
                )
                gy, gx = np.unravel_index(valid, (H, W))
                grid = np.stack([gx + 0.5, gy + 0.5], axis=1).astype(np.float32)
                x1y1 = (grid - dfl[:, :2]) * stride
                x2y2 = (grid + dfl[:, 2:]) * stride
                boxes = np.concatenate([x1y1, x2y2], axis=1)
                boxes[:, [0, 2]] *= ow / INP
                boxes[:, [1, 3]] *= oh / INP
                ab.append(boxes)
                as0.append(scores)
                ac.append(vcls)

            if not ab:
                continue

            boxes = np.concatenate(ab)
            scores = np.concatenate(as0)
            cls_ids = np.concatenate(ac)
            bb = np.stack(
                [
                    boxes[:, 0],
                    boxes[:, 1],
                    boxes[:, 2] - boxes[:, 0],
                    boxes[:, 3] - boxes[:, 1],
                ],
                1,
            )
            fb, fs, fc = [], [], []
            for cid in np.unique(cls_ids):
                if cid >= NC:
                    continue
                idx = cls_ids == cid
                ind = cv2.dnn.NMSBoxes(
                    bb[idx].tolist(), scores[idx].tolist(), CONF, NMS_IOU
                )
                if len(ind) > 0:
                    ind = np.array(ind).flatten()
                    fb.append(boxes[idx][ind])
                    fs.append(scores[idx][ind])
                    fc.append(cls_ids[idx][ind])

            if fb:
                pb = np.concatenate(fb)
                ps = np.concatenate(fs)
                pc = np.concatenate(fc)
                t, f, fn0 = match(pb, ps, pc, gb, gc)
                tp += t
                fp += f
                fn += fn0

    dt = time.time() - t0
    n = len(all_ims)
    p = tp / (tp + fp + 1e-6)
    r = tp / (tp + fn + 1e-6)
    F1 = 2 * p * r / (p + r + 1e-6)
    print(f"  {label}: {dt / n * 1000:.0f}ms/img  F1={F1:.4f}  P={p:.4f}  R={r:.4f}  TP={tp}  FP={fp}  FN={fn}")
    return F1, p, r, tp, fp, fn


# ============================================================
# 主流程
# ============================================================
if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "cal"

    if mode == "cal":
        scales = calibrate()
        # 校准完后立即用固定 scale 评估
        evaluate_fixed(scales)

    elif mode == "all":
        # 完整对比：动态 vs 固定
        scales = calibrate()
        evaluate_dynamic()
        evaluate_fixed(scales)

    elif mode == "eval_fixed":
        evaluate_fixed()

    elif mode == "eval_dynamic":
        evaluate_dynamic()

    else:
        print(f"用法: python qat_fixed_scale.py [cal|all|eval_fixed|eval_dynamic]")
