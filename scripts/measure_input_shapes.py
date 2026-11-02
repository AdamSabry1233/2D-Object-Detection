"""
Record the real network input shape each model sees for one KITTI val image,
by running each model's actual preprocessing (same predict() settings as
scripts/eval_detector.py). Writes results/input_shapes.json (a new file; no
existing result file is modified).
"""
import json
from pathlib import Path

import cv2
import torch
import torchvision.transforms.functional as TF
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMG_DIR = Path("C:/dev/datasets/kitti/images/val")
gt = json.loads((ROOT / "annotations" / "kitti_val_coco.json").read_text())
from collections import Counter
(w0, h0), _ = Counter((i["width"], i["height"]) for i in gt["images"]).most_common(1)[0]
info = next(i for i in gt["images"] if (i["width"], i["height"]) == (w0, h0))
path = IMG_DIR / info["file_name"]
out = {"image": info["file_name"], "orig_w": w0, "orig_h": h0}


def ultralytics_shape(cls_name, weights, imgsz):
    import ultralytics
    m = getattr(ultralytics, cls_name)(weights)
    kw = dict(imgsz=imgsz, conf=0.001, iou=0.7, max_det=100, device=0, verbose=False)
    img = cv2.imread(str(path))
    m.predict(img, **kw)                      # creates m.predictor
    pred, seen = m.predictor, []
    orig = pred.preprocess

    def wrap(im):
        o = orig(im)
        seen.append(tuple(o.shape))
        return o
    pred.preprocess = wrap
    m.predict(img, **kw)
    n, c, h, w = seen[0]
    return dict(input_h=h, input_w=w, scale_fill=bool(getattr(pred, "scale_fill", False)))


out["yolo11s_640"] = ultralytics_shape("YOLO", "yolo11s.pt", 640)
out["yolo11s_1280"] = ultralytics_shape("YOLO", "yolo11s.pt", 1280)
out["rtdetr-l_640"] = ultralytics_shape("RTDETR", "rtdetr-l.pt", 640)

from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
fr = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1).eval().cuda()
t = TF.to_tensor(Image.open(path).convert("RGB")).cuda()
with torch.inference_mode():
    il, _ = fr.transform([t])
h, w = il.image_sizes[0]
ph, pw = il.tensors.shape[-2:]
out["frcnn"] = dict(resized_h=h, resized_w=w, padded_h=ph, padded_w=pw)
(ROOT / "results" / "input_shapes.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
