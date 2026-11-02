"""
Render compact, report-sized versions of the 6 failure figures listed in
figures/report/ (same images, same matching at score >= 0.4 and IoU >= 0.5, as
scripts/failure_analysis.py). Each panel is cropped vertically to the band that
contains the drawn boxes so three panels fit on a page at readable size.
Output: figures/report/compact/<same file name>
"""
import json, sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from failure_analysis import match, MODELS, NAMES, GT_FILE, RUNS, IMG_DIR  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "figures" / "report"
OUT = SRC / "compact"
CONF, IOU = 0.4, 0.5
LABEL = {"yolo11s": "YOLO11s", "rtdetr-l": "RT-DETR-L", "frcnn": "Faster R-CNN"}
WIDTH_IN = 7.2


def draw_box(ax, b, color, lw, ls, label=None, ymin=0, fs=6.5):
    if label and b[2] * b[3] <= 6000:
        label = None   # labels only on large boxes; small ones overlap and are unreadable
    ax.add_patch(patches.Rectangle((b[0], b[1]), b[2], b[3], fill=False,
                                   edgecolor=color, linewidth=lw, linestyle=ls))
    if label:
        ax.text(b[0] + 1, max(b[1] - 2, ymin + 9), label, color=color, fontsize=fs, va="bottom")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gt = json.loads(GT_FILE.read_text())
    imgs = {i["file_name"]: i for i in gt["images"]}
    gts = defaultdict(list)
    for a in gt["annotations"]:
        gts[a["image_id"]].append(a)
    wanted = sorted(p.stem.split("_")[-1] + ".png" for p in SRC.glob("*.png"))
    ids = {imgs[w]["id"] for w in wanted}

    dets = {}
    for m in MODELS:
        by = defaultdict(list)
        for d in json.loads((RUNS / f"{m}_kitti_dets.json").read_text()):
            if d["image_id"] in ids and d["score"] >= CONF and d["category_id"] in NAMES:
                by[d["image_id"]].append(d)
        dets[m] = {i: match(gts[i], sorted(by[i], key=lambda d: -d["score"]), IOU) for i in ids}

    for p in sorted(SRC.glob("*.png")):
        fname = p.stem.split("_")[-1] + ".png"
        info = imgs[fname]
        iid, W, H = info["id"], info["width"], info["height"]
        im = Image.open(IMG_DIR / fname).convert("RGB")
        boxes = [g["bbox"] for g in gts[iid]]
        for m in MODELS:
            tp, fp, missed = dets[m][iid]
            boxes += [d["bbox"] for d in tp + fp] + [g["bbox"] for g in missed]
        small = [b for b in boxes if b[3] < 0.5 * H] or boxes   # ignore very tall boxes when picking the band
        ymin = max(0, min(b[1] for b in small) - 25)
        ymax = min(H, max(b[1] + b[3] for b in small) + 15)
        ymax = max(ymax, ymin + 0.3 * H)
        ymax = min(ymax, H)
        ratios = [(ymax - ymin) / W * WIDTH_IN + 0.18] * len(MODELS)
        fig, axes = plt.subplots(len(MODELS), 1, figsize=(WIDTH_IN, sum(ratios)),
                                 gridspec_kw=dict(height_ratios=ratios, hspace=0.12))
        for ax, m in zip(axes, MODELS):
            tp, fp, missed = dets[m][iid]
            ax.imshow(im)
            ax.set_xlim(0, W)
            ax.set_ylim(ymax, ymin)
            ax.axis("off")
            for g in gts[iid]:
                if g["iscrowd"]:
                    draw_box(ax, g["bbox"], "gray", 0.9, "--")
            for d in tp:
                draw_box(ax, d["bbox"], "dodgerblue", 1.0, "-")
            for d in fp:
                draw_box(ax, d["bbox"], "red", 1.3, "-", f'{NAMES[d["category_id"]]} {d["score"]:.2f}', ymin, 6.5)
            for g in missed:
                draw_box(ax, g["bbox"], "orange", 1.6, "-",
                         NAMES[g["category_id"]], ymin, 6.5)
            head = f"KITTI {fname[:-4]}: " if m == MODELS[0] else ""
            ax.set_title(f"{head}{LABEL[m]}: {len(tp)} correct, {len(fp)} false positives, {len(missed)} missed",
                         fontsize=7.5, loc="left", pad=2)
        fig.subplots_adjust(left=0.005, right=0.995, top=0.97, bottom=0.005)
        fig.savefig(OUT / p.name, dpi=220)
        plt.close(fig)
        print(f"{p.name}: crop y {ymin:.0f}-{ymax:.0f} of {H}")


if __name__ == "__main__":
    main()
