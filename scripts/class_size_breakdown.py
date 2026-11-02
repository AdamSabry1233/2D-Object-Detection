"""Per-class x object-size AP (pycocotools) from saved detections, COCO and KITTI."""
import csv, io, json
from contextlib import redirect_stdout
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("C:/dev/runs")
GT = {"coco": Path("C:/dev/datasets/coco/annotations/instances_val2017.json"),
      "kitti": ROOT / "annotations" / "kitti_val_coco.json"}
MODELS = ["yolo11s", "rtdetr-l", "frcnn"]
CATS = {1: "person", 3: "car", 8: "truck"}
SIZES = ["small", "medium", "large"]


def size_of(a):
    return "small" if a <= 32**2 else "medium" if a <= 96**2 else "large"


def fmt(v, n):
    s = "  -  " if v < 0 else f"{v:.3f}"
    return s + ("*" if n < 30 else " ")  # * = fewer than 30 GT objects, unreliable


rows = []
for ds, path in GT.items():
    with redirect_stdout(io.StringIO()):
        gt = COCO(str(path))
    img_ids = sorted(gt.getImgIds())
    n = {(c, s): 0 for c in CATS for s in SIZES}
    for a in gt.dataset["annotations"]:
        if a["category_id"] in CATS and not a["iscrowd"]:
            n[(a["category_id"], size_of(a["area"]))] += 1
    total = sum(n.values())
    print(f"\n=== {ds.upper()} === GT share by size (person/car/truck, n={total}): " +
          ", ".join(f"{s} {sum(n[(c, s)] for c in CATS) / total:.0%}" for s in SIZES))
    print(f"{'class':>7} {'model':>9} {'AP':>7} {'APs':>7} {'APm':>7} {'APl':>7}   n small/med/large")
    for m in MODELS:
        dets = [d for d in json.loads((RUNS / f"{m}_{ds}_dets.json").read_text())
                if d["category_id"] in CATS]
        dt = gt.loadRes(dets)
        for c, cname in CATS.items():
            with redirect_stdout(io.StringIO()):
                E = COCOeval(gt, dt, "bbox")
                E.params.imgIds, E.params.catIds = img_ids, [c]
                E.evaluate(); E.accumulate(); E.summarize()
            st = E.stats
            ns = [n[(c, s)] for s in SIZES]
            rows.append(dict(dataset=ds, model=m, cls=cname, AP=round(st[0], 4),
                             AP_small=round(st[3], 4), AP_medium=round(st[4], 4),
                             AP_large=round(st[5], 4), n_small=ns[0], n_medium=ns[1], n_large=ns[2]))
            print(f"{cname:>7} {m:>9} {fmt(st[0], 99)} {fmt(st[3], ns[0])} {fmt(st[4], ns[1])} "
                  f"{fmt(st[5], ns[2])}   {ns[0]}/{ns[1]}/{ns[2]}")

with open(ROOT / "results" / "class_size_ap.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print("\nWrote results/class_size_ap.csv")