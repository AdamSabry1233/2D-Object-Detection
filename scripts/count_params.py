"""Count parameters and FP32 weight size of the three evaluated detectors."""
import json
from pathlib import Path

from ultralytics import YOLO, RTDETR
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights)

ROOT = Path(__file__).resolve().parent.parent


def n_params(m):
    return sum(p.numel() for p in m.parameters())


models = {
    "yolo11s": YOLO("yolo11s.pt").model,
    "rtdetr-l": RTDETR("rtdetr-l.pt").model,
    "frcnn": fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.COCO_V1),
}
out = {}
for name, m in models.items():
    p = n_params(m)
    out[name] = {"params": p, "params_M": round(p / 1e6, 2),
                 "fp32_weights_MB": round(p * 4 / 1e6, 1)}
    print(f"{name}: {p:,} params ({p / 1e6:.1f} M), FP32 weights ~{p * 4 / 1e6:.0f} MB")

(ROOT / "results" / "params").mkdir(parents=True, exist_ok=True)
(ROOT / "results" / "params" / "model_params.json").write_text(json.dumps(out, indent=2))
print("Wrote results/params/model_params.json")