"""板端评测脚本"""
import os, sys; os.chdir('/home/sunrise/yolo_deploy')
import numpy as np, cv2, time, glob
from hobot_dnn import pyeasy_dnn as dnn

MODEL = sys.argv[1] if len(sys.argv) > 1 else '/home/sunrise/yolo_deploy/yolov8n_2cls_bias0_bayese_640x640_nv12.bin'
VAL_IMG = 'val/images'; VAL_LAB = 'val/labels'
NC = 2; CONF = 0.25; NMS_IOU = 0.45; REG_MAX = 16; INP = 640

def load_gt(lp, iw, ih):
    bs, cs = [], []
    if not os.path.exists(lp): return bs, cs
    for line in open(lp).read().strip().splitlines():
        p = line.split()
        if len(p) < 5: continue
        c, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
        bs.append([int((cx - w / 2) * iw), int((cy - h / 2) * ih), int((cx + w / 2) * iw), int((cy + h / 2) * ih)])
        cs.append(c)
    return bs, cs

def iou(b1, b2):
    x1, y1, x2, y2 = max(b1[0], b2[0]), max(b1[1], b2[1]), min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((b1[2] - b1[0]) * (b1[3] - b1[1]) + (b2[2] - b2[0]) * (b2[3] - b2[1]) - inter + 1e-6)

def match(pb, ps, pc, gb, gc):
    gm = [False] * len(gb); dm = [False] * len(pb)
    for i in np.argsort(ps)[::-1]:
        bi, bj = 0, -1
        for j, (b, c) in enumerate(zip(gb, gc)):
            if gm[j] or c != pc[i]: continue
            v = iou(pb[i], b)
            if v > bi: bi, bj = v, j
        if bi >= 0.5: dm[i] = True; gm[bj] = True
    tp = sum(dm); fp = len(pb) - tp; fn = len(gb) - sum(gm)
    return tp, fp, fn

def bgr_to_nv12(img):
    h, w = img.shape[:2]
    s = min(INP / h, INP / w); nh, nw = int(h * s), int(w * s)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.full((INP, INP, 3), 114, np.uint8)
    pt, pl = (INP - nh) // 2, (INP - nw) // 2
    canvas[pt:pt + nh, pl:pl + nw] = resized
    yuv = cv2.cvtColor(canvas, cv2.COLOR_BGR2YUV_I420)
    y = yuv[:INP, :]; u = yuv[INP:INP + INP // 4, :]; v = yuv[INP + INP // 4:, :]
    uv = np.zeros((INP // 2, INP), np.uint8); uv[0::2, :] = u; uv[1::2, :] = v
    return np.concatenate([y, uv], axis=0), s, pl, pt

def postprocess(outputs, ow, oh, scale, pad_left, pad_top):
    strides = [8, 16, 32]; ab, as0, ac = [], [], []
    for si, s in enumerate(strides):
        bbox_buf = np.array(outputs[si * 2].buffer, copy=False).astype(np.float32)
        cls_buf = np.array(outputs[si * 2 + 1].buffer, copy=False).astype(np.float32)
        H, W = cls_buf.shape[2], cls_buf.shape[3]
        cls = cls_buf.transpose(0, 2, 3, 1).reshape(-1, NC)
        scores = 1 / (1 + np.exp(-cls)); mx = scores.max(1); mc = scores.argmax(1)
        valid = np.flatnonzero(mx > CONF)
        if len(valid) == 0: continue
        box = bbox_buf.transpose(0, 2, 3, 1).reshape(-1, REG_MAX * 4)
        b = box[valid]
        e = np.exp(b.reshape(-1, 4, REG_MAX) - b.reshape(-1, 4, REG_MAX).max(axis=2, keepdims=True))
        dfl = (e / e.sum(axis=2, keepdims=True) * np.arange(REG_MAX)).sum(axis=2)
        gy, gx = np.unravel_index(valid, (H, W))
        grid = np.stack([gx + 0.5, gy + 0.5], axis=1).astype(np.float32)
        x1y1 = (grid - dfl[:, :2]) * s; x2y2 = (grid + dfl[:, 2:]) * s
        boxes = np.concatenate([x1y1, x2y2], axis=1)
        boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_left) / scale
        boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_top) / scale
        ab.append(boxes); as0.append(mx[valid]); ac.append(mc[valid])
    if not ab: return [], [], []
    boxes = np.concatenate(ab); scores = np.concatenate(as0); cls_ids = np.concatenate(ac)
    bb = np.stack([boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
    fb, fs, fc = [], [], []
    for cid in np.unique(cls_ids):
        idx = cls_ids == cid
        ind = cv2.dnn.NMSBoxes(bb[idx].tolist(), scores[idx].tolist(), CONF, NMS_IOU)
        if len(ind) > 0:
            ind = np.array(ind).flatten()
            fb.append(boxes[idx][ind]); fs.append(scores[idx][ind]); fc.append(cls_ids[idx][ind])
    if not fb: return [], [], []
    return np.concatenate(fb), np.concatenate(fs), np.concatenate(fc)

models = dnn.load(MODEL); mdl = models[0] if isinstance(models, list) else models
img_paths = sorted(glob.glob(os.path.join(VAL_IMG, '*.jpg')))
print(f'图片数: {len(img_paths)} 模型: {os.path.basename(MODEL)}')
tp = fp = fn = 0; total_t = 0
for i, ip in enumerate(img_paths):
    bn = os.path.splitext(os.path.basename(ip))[0]
    lp = os.path.join(VAL_LAB, bn + '.txt')
    img = cv2.imread(ip); oh, ow = img.shape[:2]; gb, gc = load_gt(lp, ow, oh)
    t0 = time.time(); nv12, sc, pl, pt = bgr_to_nv12(img)
    outputs = mdl.forward(nv12.flatten()); t1 = time.time()
    pb, ps, pc = postprocess(outputs, ow, oh, sc, pl, pt); t2 = time.time()
    t, f, fn0 = match(pb, ps, pc, gb, gc)
    tp += t; fp += f; fn += fn0; total_t += (t2 - t0) * 1000
    if (i + 1) % 20 == 0:
        print(f'{i + 1}/{len(img_paths)} gt={len(gb)} pred={len(pb)} tp={t}')

n = len(img_paths); avg_t = total_t / n if n > 0 else 0
p = tp / (tp + fp + 1e-6); r = tp / (tp + fn + 1e-6); F1 = 2 * p * r / (p + r + 1e-6)
print(f'\n{os.path.basename(MODEL)}: {avg_t:.0f}ms F1={F1:.4f} TP={tp} FP={fp} FN={fn}')
