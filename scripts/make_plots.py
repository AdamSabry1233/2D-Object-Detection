"""Make report figures (COCO vs KITTI, per-class, speed-accuracy) from results/*.json."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"font.size": 8.5, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
                     "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8})

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

MODELS = ["yolo11s", "rtdetr-l", "frcnn"]
LABEL = {"yolo11s": "YOLO11s", "rtdetr-l": "RT-DETR-L", "frcnn": "Faster R-CNN"}
DS_COLOR = {"coco": "#4C78A8", "kitti": "#F58518"}
MODEL_COLOR = {"yolo11s": "#54A24B", "rtdetr-l": "#B279A2", "frcnn": "#E45756"}

R = {(m, ds): json.loads((ROOT / "results" / f"{m}_{ds}.json").read_text())
     for m in MODELS for ds in ["coco", "kitti"]}


def m3(m, ds):
    return R[(m, ds)]["metrics"]["3class_person_car_truck"]


def bar_labels(ax, bars, fs=7.5):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=fs)


# ---- Figure 1: overall 3-class AP, COCO vs KITTI ----
fig, ax = plt.subplots(figsize=(6.0, 3.5))
x = np.arange(len(MODELS))
w = 0.36
for i, ds in enumerate(["coco", "kitti"]):
    vals = [m3(m, ds)["stats"]["AP"] for m in MODELS]
    bars = ax.bar(x + (i - 0.5) * w, vals, w, label=ds.upper(), color=DS_COLOR[ds])
    bar_labels(ax, bars)
for j, m in enumerate(MODELS):
    c, k = m3(m, "coco")["stats"]["AP"], m3(m, "kitti")["stats"]["AP"]
    ax.text(j, max(c, k) + 0.045, f"{(k - c) / c:+.0%}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([LABEL[m] for m in MODELS])
ax.set_ylabel("AP@[.50:.95] (person, car, truck)")
ax.set_ylim(0, 0.65)
ax.set_title("COCO-pretrained detectors: COCO val2017 vs. KITTI val (no retraining)", fontsize=9)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig1_coco_vs_kitti_ap.png", dpi=200)
plt.close(fig)

# ---- Figure 2: per-class AP, COCO vs KITTI ----
classes = ["person", "car", "truck"]
fig, axes = plt.subplots(1, 3, figsize=(7.4, 3.3), sharey=True)
for ax, cls in zip(axes, classes):
    for i, ds in enumerate(["coco", "kitti"]):
        vals = [m3(m, ds)["per_class_AP"][cls] for m in MODELS]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=ds.upper(), color=DS_COLOR[ds])
        bar_labels(ax, bars, 6.3)
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[m] for m in MODELS], rotation=20, fontsize=7)
    ax.set_title(cls + (" (n=223 in KITTI, noisy)" if cls == "truck" else ""))
axes[0].set_ylabel("AP@[.50:.95]")
axes[0].set_ylim(0, 0.75)
axes[0].legend()
fig.suptitle("Per-class AP (3-class): person and truck degrade; car is flat overall", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "fig2_per_class_ap.png", dpi=200)
plt.close(fig)

# ---- Figure 3: speed-accuracy tradeoff ----
fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)
for ax, ds in zip(axes, ["coco", "kitti"]):
    for m in MODELS:
        lat = R[(m, ds)]["latency_ms"]["mean"]
        ap = m3(m, ds)["stats"]["AP"]
        ax.scatter(lat, ap, s=70, color=MODEL_COLOR[m], zorder=3)
        ax.annotate(f"{LABEL[m]}\n{lat:.0f} ms, {R[(m, ds)]['fps']:.0f} FPS", (lat, ap),
                    textcoords="offset points", xytext=(8, -4), fontsize=7.5)
    ax.set_xlabel("mean latency per image (ms)")
    ax.set_title(ds.upper())
    ax.set_xlim(0, 145 if ds == "coco" else 100)
    ax.set_ylim(0.22, 0.56)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("AP@[.50:.95] (person, car, truck)")
fig.suptitle("Speed-accuracy tradeoff (RTX 4050 Laptop; latency = preprocess + model + NMS)", fontsize=9.5)
fig.tight_layout()
fig.savefig(OUT / "fig3_speed_accuracy.png", dpi=200)
plt.close(fig)

print("Saved fig1_coco_vs_kitti_ap.png, fig2_per_class_ap.png, fig3_speed_accuracy.png to", OUT)