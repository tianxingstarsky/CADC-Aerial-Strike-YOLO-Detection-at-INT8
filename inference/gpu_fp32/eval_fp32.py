"""诊断2：详细分析每张图的 TP/FP/FN"""
import os, warnings
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
warnings.filterwarnings("ignore")

import torch, numpy as np, cv2, types
from pathlib import Path
from ultralytics import YOLO
from ultralytics.nn.modules.head import Detect

PROJECT = Path(r"F:\RDKX5投弹")
DEVICE = "cuda:0"
BEST_PT = PROJECT / "runs/yolov8n_2cls/weights/best.pt"
VAL_IMG = PROJECT / "val/images"
VAL_LAB = PROJECT / "val/labels"
NC = 2; REG_MAX = 16; INP = 640

m = YOLO(str(BEST_PT))
model = m.model
model.to(DEVICE).eval()

detect = None
for child in model.modules():
    if isinstance(child, Detect):
        detect = child
        break
def Df(self, _x):
    r = []
    for i in range(self.nl):
        r.append(self.cv2[i](_x[i]))
        r.append(self.cv3[i](_x[i]))
    return r
detect.forward = types.MethodType(Df, detect)

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

def iou(b1,b2):
    x1,y1,x2,y2=max(b1[0],b2[0]),max(b1[1],b2[1]),min(b1[2],b2[2]),min(b1[3],b2[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/((b1[2]-b1[0])*(b1[3]-b1[1])+(b2[2]-b2[0])*(b2[3]-b2[1])-inter+1e-6)

def match(pb,ps,pc,gb,gc):
    gm=[False]*len(gb); dm=[False]*len(pb)
    for i in np.argsort(ps)[::-1]:
        bi,bj=0,-1
        for j,(b,c) in enumerate(zip(gb,gc)):
            if gm[j] or c!=pc[i]: continue
            v=iou(pb[i],b)
            if v>bi: bi,bj=v,j
        if bi>=0.5: dm[i]=True; gm[bj]=True
    tp=sum(dm); fp=len(pb)-tp; fn=len(gb)-sum(gm)
    return tp,fp,fn

def detect_frame(tensor, model):
    raw = model(tensor)
    strides=[8,16,32]
    ab,as0,ac=[],[],[]
    for si,s in enumerate(strides):
        cls_buf=raw[si*2+1]
        cls=cls_buf.permute(0,2,3,1).reshape(-1,2).cpu().numpy()
        scores=1/(1+np.exp(-cls)); mx=scores.max(1); mc=scores.argmax(1)
        valid=np.flatnonzero(mx>0.25)
        if len(valid)==0: continue
        bbox_buf=raw[si*2]; H,W=cls_buf.shape[2],cls_buf.shape[3]
        box=bbox_buf.permute(0,2,3,1).reshape(-1,REG_MAX*4).cpu().numpy()
        b=box[valid]; bmax=b.reshape(-1,4,REG_MAX).max(axis=2,keepdims=True)
        e=np.exp(b.reshape(-1,4,REG_MAX)-bmax)
        dfl=(e/e.sum(axis=2,keepdims=True)*np.arange(REG_MAX)).sum(axis=2)
        gy,gx=np.unravel_index(valid,(H,W))
        grid=np.stack([gx+0.5,gy+0.5],axis=1).astype(np.float32)
        x1y1=(grid-dfl[:,:2])*s; x2y2=(grid+dfl[:,2:])*s
        boxes=np.concatenate([x1y1,x2y2],axis=1)
        ab.append(boxes); as0.append(mx[valid]); ac.append(mc[valid])
    if not ab: return [],[],[]
    boxes=np.concatenate(ab); scores=np.concatenate(as0); cls_ids=np.concatenate(ac)
    bb=np.stack([boxes[:,0],boxes[:,1],boxes[:,2]-boxes[:,0],boxes[:,3]-boxes[:,1]],1)
    fb,fs,fc=[],[],[]
    for cid in np.unique(cls_ids):
        if cid>=NC: continue
        idx=cls_ids==cid
        ind=cv2.dnn.NMSBoxes(bb[idx].tolist(),scores[idx].tolist(),0.25,0.45)
        if len(ind)>0:
            ind=np.array(ind).flatten()
            fb.append(boxes[idx][ind]); fs.append(scores[idx][ind]); fc.append(cls_ids[idx][ind])
    if not fb: return [],[],[]
    return np.concatenate(fb),np.concatenate(fs),np.concatenate(fc)

all_ims = sorted(list(VAL_IMG.glob("*.jpg")))[:50]
total_tp=total_fp=total_fn=0

with torch.no_grad():
    for idx, ip in enumerate(all_ims):
        img = cv2.imread(str(ip)); oh, ow = img.shape[:2]
        lp = VAL_LAB / (ip.stem + ".txt")
        gb, gc = load_gt(lp, ow, oh)

        tensor = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = cv2.resize(tensor, (640, 640))
        tensor = torch.from_numpy(tensor).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
        pb, ps, pc = detect_frame(tensor, model)

        # 映射回原图
        if len(pb) > 0:
            pb = pb.copy()
            pb[:, [0, 2]] *= ow / INP
            pb[:, [1, 3]] *= oh / INP

        tp, fp, fn = match(pb, ps, pc, gb, gc)
        total_tp += tp; total_fp += fp; total_fn += fn

        if tp == 0 and len(gb) > 0:
            std_scores = 1/(1+np.exp(-np.array([raw_sig for raw_sig in []])))
            raw = model(tensor)
            # 看原始 logits 最高值
            max_logit = -999
            for si in range(3):
                cls_buf = raw[si*2+1]
                cls_f = cls_buf.permute(0,2,3,1).reshape(-1,2).cpu().numpy()
                max_logit = max(max_logit, cls_f.max())
            print(f"  {ip.name} ({ow}x{oh}): gt={len(gb)} pred={len(pb)} tp={tp} fp={fp} fn={fn}  max_logit={max_logit:.2f} max_sigmoid={1/(1+np.exp(-max_logit)):.4f}")

n = len(all_ims)
p = total_tp/(total_tp+total_fp+1e-6)
r = total_tp/(total_tp+total_fn+1e-6)
F1 = 2*p*r/(p+r+1e-6)
print(f"\n总计: F1={F1:.4f} P={p:.4f} R={r:.4f} TP={total_tp} FP={total_fp} FN={total_fn}")
