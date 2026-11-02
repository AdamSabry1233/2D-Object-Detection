"""
Evaluate a COCO-pretrained detector on COCO val2017 or the KITTI val subset,
scoring with pycocotools. Writes metrics, latency, memory, and config to JSON.
"""
import argparse, json, sys, time
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = Path("C:/dev/runs")  # raw detections go outside OneDrive

DATASETS = {
    "coco": dict(gt=Path("C:/dev/datasets/coco/annotations/instances_val2017.json"),
                 img_dir=Path("C:/dev/datasets/coco/val2017")),
    "kitti": dict(gt=ROOT / "annotations" / "kitti_val_coco.json",
                  img_dir=Path("C:/dev/datasets/kitti/images/val")),
}
EVAL_CATS = [1, 3, 8]  # person, car, truck (original COCO category ids)

# Ultralytics outputs class index 0-79; COCO annotations use ids 1-90 with gaps.
COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
assert len(COCO80_TO_91) == 80


class UltralyticsDet:
    """YOLO11 / RT-DETR via Ultralytics. prepare() = decode image, infer() = timed."""
    def __init__(self, cls_name, weights, imgsz, conf, iou, max_det):
        import ultralytics
        self.model = getattr(ultralytics, cls_name)(weights)
        self.kw = dict(imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
                       device=0, verbose=False)

    def prepare(self, path):
        return cv2.imread(str(path))  # BGR numpy array, which Ultralytics expects

    def infer(self, img):
        r = self.model.predict(img, **self.kw)[0]
        b = r.boxes
        cats = np.array([COCO80_TO_91[c] for c in b.cls.cpu().numpy().astype(int)])
        return b.xyxy.cpu().numpy(), b.conf.cpu().numpy(), cats


class FasterRCNN:
    """torchvision Faster R-CNN R50-FPN, COCO_V1 weights (resizes internally to min 800)."""
    def __init__(self, conf, iou, max_det):
        from torchvision.models.detection import (
            fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)
        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1,
            box_score_thresh=conf, box_nms_thresh=iou,
            box_detections_per_img=max_det).eval().cuda()

    def prepare(self, path):
        import torchvision.transforms.functional as TF
        return TF.to_tensor(Image.open(path).convert("RGB"))

    @torch.inference_mode()
    def infer(self, t):
        o = self.model([t.cuda()])[0]
        return o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy()


def run_eval(gt, dets, img_ids, cat_ids):
    E = COCOeval(gt, gt.loadRes(dets), "bbox")
    E.params.imgIds, E.params.catIds = img_ids, cat_ids
    E.evaluate(); E.accumulate(); E.summarize()
    names = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
             "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
    stats = {n: round(float(v), 4) for n, v in zip(names, E.stats)}
    prec = E.eval["precision"]  # [IoU thr, recall, class, area, maxDets]
    per_class = {}
    for k, c in enumerate(cat_ids):
        p = prec[:, :, k, 0, 2]
        p = p[p > -1]
        per_class[gt.cats[c]["name"]] = round(float(p.mean()), 4) if p.size else None
    return stats, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["yolo11s", "rtdetr-l", "frcnn"])
    ap.add_argument("--dataset", required=True, choices=["coco", "kitti"])
    ap.add_argument("--imgsz", type=int, default=640, help="Ultralytics models only")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=None, help="NMS IoU (default per framework)")
    ap.add_argument("--max-det", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="only first N images (smoke test)")
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    if args.model == "frcnn":
        iou = args.iou if args.iou is not None else 0.5
        det = FasterRCNN(args.conf, iou, args.max_det)
        cfg = dict(checkpoint="torchvision FasterRCNN_ResNet50_FPN_Weights.COCO_V1",
                   input_size="internal resize: min side 800, max side 1333")
    else:
        iou = args.iou if args.iou is not None else 0.7
        cls_name, weights = ("YOLO", "yolo11s.pt") if args.model == "yolo11s" else ("RTDETR", "rtdetr-l.pt")
        det = UltralyticsDet(cls_name, weights, args.imgsz, args.conf, iou, args.max_det)
        cfg = dict(checkpoint=weights, input_size=f"imgsz={args.imgsz} (letterboxed)")
    cfg.update(conf_thresh=args.conf, nms_iou=(None if args.model == "rtdetr-l" else iou),
               max_det=args.max_det)

    ds = DATASETS[args.dataset]
    gt = COCO(str(ds["gt"]))
    img_ids = sorted(gt.getImgIds())
    if args.limit:
        img_ids = img_ids[:args.limit]

    torch.cuda.reset_peak_memory_stats()
    dets, lat = [], []
    for i, iid in enumerate(img_ids):
        inp = det.prepare(ds["img_dir"] / gt.imgs[iid]["file_name"])
        torch.cuda.synchronize(); t0 = time.perf_counter()
        boxes, scores, cats = det.infer(inp)
        torch.cuda.synchronize(); ms = (time.perf_counter() - t0) * 1000
        if i >= args.warmup:
            lat.append(ms)
        for (x1, y1, x2, y2), s, c in zip(boxes, scores, cats):
            dets.append({"image_id": iid, "category_id": int(c), "score": round(float(s), 4),
                         "bbox": [round(float(x1), 2), round(float(y1), 2),
                                  round(float(x2 - x1), 2), round(float(y2 - y1), 2)]})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(img_ids)} images", flush=True)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    tag = (f"{args.model}_{args.dataset}"
           + (f"_imgsz{args.imgsz}" if args.model != "frcnn" and args.imgsz != 640 else "")
           + (f"_limit{args.limit}" if args.limit else ""))
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / f"{tag}_dets.json").write_text(json.dumps(dets))

    metrics = {}
    if args.dataset == "coco":
        s, pc = run_eval(gt, dets, img_ids, sorted(gt.getCatIds()))
        metrics["coco_80class"] = {"stats": s, "per_class_AP": pc}
        kitti_dets = [d for d in dets if d["category_id"] in EVAL_CATS]
    else:
        kitti_dets = [d for d in dets if d["category_id"] in EVAL_CATS]
    s, pc = run_eval(gt, kitti_dets, img_ids, EVAL_CATS)
    metrics["3class_person_car_truck"] = {"stats": s, "per_class_AP": pc}

    out = dict(model=args.model, dataset=args.dataset, n_images=len(img_ids), config=cfg,
               metrics=metrics,
               latency_ms=dict(mean=round(float(np.mean(lat)), 2),
                               median=round(float(np.median(lat)), 2),
                               p95=round(float(np.percentile(lat, 95)), 2)),
               fps=round(1000 / float(np.mean(lat)), 1),
               peak_gpu_mem_gb=round(peak_gb, 3),
               hardware=torch.cuda.get_device_name(0),
               versions={p: version(p) for p in
                         ["torch", "torchvision", "ultralytics", "pycocotools", "numpy"]},
               python=sys.version.split()[0],
               command="python " + " ".join(sys.argv))
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{tag}.json").write_text(json.dumps(out, indent=2))
    print(f"\nSaved results/{tag}.json")
    print(f"Latency mean {out['latency_ms']['mean']} ms  |  {out['fps']} FPS  |  peak mem {peak_gb:.2f} GB")


if __name__ == "__main__":
    main()