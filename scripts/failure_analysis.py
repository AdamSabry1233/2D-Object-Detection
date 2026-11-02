"""
Failure analysis on the KITTI val subset.
Reads saved detections from C:/dev/runs (no inference is rerun), matches them to
ground truth, ranks images by failure type, and draws GT vs. predictions.
"""
import argparse, csv, json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RUNS = Path("C:/dev/runs")
IMG_DIR = Path("C:/dev/datasets/kitti/images/val")
GT_FILE = ROOT / "annotations" / "kitti_val_coco.json"
OUT = ROOT / "figures" / "failures"
MODELS = ["yolo11s", "rtdetr-l", "frcnn"]
NAMES = {1: "person", 3: "car", 8: "truck"}


def overlap(a, b):
    """IoU and raw intersection area of two [x, y, w, h] boxes."""
    iw = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return (inter / union if union > 0 else 0.0), inter


def match(gts, dets, iou_thr):
    """Greedy match (highest score first). Returns (true pos, false pos, missed GT)."""
    normal = [g for g in gts if not g["iscrowd"]]
    ignore = [g for g in gts if g["iscrowd"]]
    used, tp, fp = set(), [], []
    for d in dets:
        best, best_iou = None, iou_thr
        for gi, g in enumerate(normal):
            if gi in used or g["category_id"] != d["category_id"]:
                continue
            v, _ = overlap(d["bbox"], g["bbox"])
            if v >= best_iou:
                best, best_iou = gi, v
        if best is not None:
            used.add(best)
            tp.append(d)
            continue
        area = max(d["bbox"][2] * d["bbox"][3], 1e-9)
        if any(g["category_id"] == d["category_id"]
               and overlap(d["bbox"], g["bbox"])[1] / area >= 0.5 for g in ignore):
            continue  # inside an ignore region: counts as neither TP nor FP
        fp.append(d)
    missed = [g for gi, g in enumerate(normal) if gi not in used]
    return tp, fp, missed


def bucket(area):
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def box(ax, b, color, lw, ls, label=None):
    ax.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], fill=False,
                                   edgecolor=color, linewidth=lw, linestyle=ls))
    if label:
        ax.text(b[0], b[1] - 2, label, color=color, fontsize=7, va="bottom")


def draw(iid, cat, rank, results, imgs, gts, args):
    info = imgs[iid]
    im = Image.open(IMG_DIR / info["file_name"]).convert("RGB")
    fig, axes = plt.subplots(len(MODELS), 1, figsize=(13, 11))
    for ax, m in zip(axes, MODELS):
        tp, fp, missed = results[m][iid]
        ax.imshow(im)
        ax.axis("off")
        for g in gts[iid]:
            if g["iscrowd"]:
                box(ax, g["bbox"], "gray", 1.0, "--")
        for d in tp:
            box(ax, d["bbox"], "dodgerblue", 1.2, "-")
        for d in fp:
            box(ax, d["bbox"], "red", 1.8, "-", f'{NAMES[d["category_id"]]} {d["score"]:.2f}')
        for g in missed:
            box(ax, g["bbox"], "orange", 2.2, "-", NAMES[g["category_id"]])
        ax.set_title(f"{m}: {len(tp)} correct, {len(fp)} false positives, {len(missed)} missed",
                     fontsize=11, loc="left")
    handles = [Line2D([0], [0], color="dodgerblue", lw=2, label="correct detection"),
               Line2D([0], [0], color="red", lw=2, label="false positive (score shown)"),
               Line2D([0], [0], color="orange", lw=3, label="missed ground truth"),
               Line2D([0], [0], color="gray", lw=2, ls="--", label="ignore region")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=9)
    fig.suptitle(f"KITTI {info['file_name']}  |  score >= {args.conf}, IoU >= {args.iou}  |  {cat}",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    path = OUT / f"{cat}_{rank}_{info['file_name'][:-4]}.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conf", type=float, default=0.4, help="score threshold for matching/drawing")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--per-category", type=int, default=4)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    gt = json.loads(GT_FILE.read_text())
    imgs = {i["id"]: i for i in gt["images"]}
    gts = defaultdict(list)
    for a in gt["annotations"]:
        gts[a["image_id"]].append(a)

    results = {}
    for m in MODELS:
        by_img = defaultdict(list)
        for d in json.loads((RUNS / f"{m}_kitti_dets.json").read_text()):
            if d["score"] >= args.conf and d["category_id"] in NAMES:
                by_img[d["image_id"]].append(d)
        results[m] = {iid: match(gts[iid], sorted(by_img[iid], key=lambda d: -d["score"]), args.iou)
                      for iid in imgs}

    # ---- 1. miss rates by size and class ----
    tot = Counter()
    for iid in imgs:
        for g in gts[iid]:
            if not g["iscrowd"]:
                tot[bucket(g["area"])] += 1
                tot[NAMES[g["category_id"]]] += 1
    cols = ["small", "medium", "large", "person", "car", "truck"]
    print(f"Scored GT objects: { {c: tot[c] for c in cols} }")
    print(f"\nMiss rate (fraction of GT with no matching detection), score >= {args.conf}, IoU >= {args.iou}")
    print(f"{'model':>10} " + " ".join(f"{c:>8}" for c in cols))
    rate_rows = []
    for m in MODELS:
        mis = Counter()
        for iid in imgs:
            for g in results[m][iid][2]:
                mis[bucket(g["area"])] += 1
                mis[NAMES[g["category_id"]]] += 1
        row = {"model": m, **{c: round(mis[c] / tot[c], 3) for c in cols}}
        rate_rows.append(row)
        print(f"{m:>10} " + " ".join(f"{row[c]:>8}" for c in cols))
    with open(OUT / "miss_rates.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model"] + cols)
        w.writeheader()
        w.writerows(rate_rows)

    # ---- 2. rank and draw failure images ----
    miss = {i: {m: len(results[m][i][2]) for m in MODELS} for i in imgs}
    fps = {i: {m: len(results[m][i][1]) for m in MODELS} for i in imgs}
    keys = {
        "missed_by_all": lambda i: (min(miss[i].values()), sum(miss[i].values())),
        "disagreement": lambda i: (max(miss[i].values()) - min(miss[i].values()), sum(miss[i].values())),
        "false_positives": lambda i: sum(fps[i].values()),
    }
    used_ids, index = set(), []
    for cat, key in keys.items():
        n = 0
        for iid in sorted(imgs, key=key, reverse=True):
            if iid in used_ids:
                continue
            used_ids.add(iid)
            n += 1
            fname = draw(iid, cat, n, results, imgs, gts, args)
            index.append({"figure": fname, "category": cat, "image": imgs[iid]["file_name"],
                          **{f"missed_{m}": miss[iid][m] for m in MODELS},
                          **{f"fp_{m}": fps[iid][m] for m in MODELS}})
            if n >= args.per_category:
                break
    with open(OUT / "index.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(index[0].keys()))
        w.writeheader()
        w.writerows(index)
    print(f"\nSaved {len(index)} figures + index.csv + miss_rates.csv to {OUT}")


if __name__ == "__main__":
    main()