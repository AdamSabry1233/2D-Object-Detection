# 2D Object Detection: COCO-pretrained detectors on COCO and KITTI (CMPE-249, Track A)

This repo evaluates three COCO-pretrained 2D detectors, without any retraining, on COCO val2017 and on the KITTI val split. It measures accuracy (pycocotools), latency, and GPU memory, and compares how well each detector generalizes from COCO to driving scenes.

The report is in `report/CMPE249_HW_2D_Detection_TrackA.pdf`. Every number in the report is computed from the files in `results/` by `scripts/build_report.py` and `scripts/claims_check.py`.

## Models

| Name in code | Family | Framework | Checkpoint |
|---|---|---|---|
| `yolo11s` | YOLO11 (anchor-free CNN, NMS) | Ultralytics | `yolo11s.pt` |
| `rtdetr-l` | RT-DETR (query-based transformer, no NMS) | Ultralytics | `rtdetr-l.pt` |
| `frcnn` | Faster R-CNN R50-FPN (two-stage, anchor-based) | torchvision | `FasterRCNN_ResNet50_FPN_Weights.COCO_V1` |

Parameter counts: `results/params/model_params.json` (made by `scripts/count_params.py`).

## Hardware and software

- GPU: NVIDIA GeForce RTX 4050 Laptop, 6 GB, power-limited to 45 W; driver 610.60
- RAM: 15.3 GB; Windows 11, run from Git Bash
- Software versions are recorded in every `results/*.json` under `versions` (torch, torchvision, ultralytics, pycocotools, numpy) and `python`. `requirements.txt` has the full `pip freeze`.

## Setup

```bash
python -m venv /c/dev/det
source /c/dev/det/Scripts/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics pycocotools
```

## Data

| Dataset | Source | Local path | Images |
|---|---|---|---|
| COCO val2017 | https://cocodataset.org/#download (`val2017.zip`, `annotations_trainval2017.zip`) | `C:/dev/datasets/coco` | 5,000 |
| KITTI val | Ultralytics repackaging of KITTI 2D object detection, https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip (see `kitti.yaml`) | `C:/dev/datasets/kitti` | 1,496 |

Datasets, weights and raw detections (`C:/dev/runs/*_dets.json`) are not in this repo because of their size. The converted KITTI ground truth is in `annotations/kitti_val_coco.json`.

### KITTI to COCO label mapping

`scripts/convert_kitti_to_coco.py` converts the YOLO-format KITTI labels to a COCO-format JSON that uses COCO category ids, so pycocotools scores both datasets the same way. The cross-dataset comparison uses three classes: person (COCO id 1), car (3), truck (8).

| KITTI class | Mapped to | Reason |
|---|---|---|
| car | car (3) | direct match |
| van | car (3) | COCO has no van class |
| truck | truck (8) | direct match |
| pedestrian | person (1) | direct match |
| person_sitting | person (1), `iscrowd=1` (ignore region) | not scored, so these are neither hits nor false positives |
| cyclist | person (1), `iscrowd=1` (ignore region) | the box covers rider plus bicycle, while COCO models predict person and bicycle separately, so scoring it would create artificial false positives |
| tram, misc | dropped | no matching class among the three evaluated |

## Evaluation protocol

- Scoring: pycocotools `COCOeval`, bbox. On COCO we report both the standard 80-class result and the 3-class result. On KITTI we report the 3-class result.
- Confidence threshold 0.001, at most 100 detections per image.
- NMS IoU: 0.7 for YOLO11s, 0.5 for Faster R-CNN, none for RT-DETR-L.
- Input handling differs by model. YOLO11s (`imgsz=640`) is letterboxed with an aspect-preserving rectangle and minimal padding. RT-DETR-L (`imgsz=640`) is **not** letterboxed: Ultralytics' RT-DETR predictor sets `scale_fill = True`, so every image is stretched to 640x640 with no padding and the aspect ratio is not preserved. Faster R-CNN uses torchvision's internal resize (min side 800, max side 1333). The three models therefore see KITTI at three different effective inputs, which confounds every pairwise comparison. The measured input shapes are in `results/input_shapes.json` (made by `scripts/measure_input_shapes.py`). The `input_size` field in the RT-DETR-L result JSONs says "letterboxed", which is inaccurate; result files were not modified.
- Latency: mean per-image time of preprocessing + model + NMS + copy to CPU (`predict()` for Ultralytics, forward pass plus copy for torchvision), after 10 warmup images. Image decoding is not timed. Measured on a 45 W power-limited laptop GPU, so small differences are noise.
- Peak GPU memory: `torch.cuda.max_memory_allocated()` (tensor allocations only; excludes the CUDA context).
- Ultralytics outputs class ids 0-79; the code maps them to COCO's 1-90 ids (`COCO80_TO_91` in `scripts/eval_detector.py`).

## Repo layout

```
scripts/        evaluation, conversion, analysis, plotting and report-building code
annotations/    kitti_val_coco.json (KITTI val in COCO format)
results/        one JSON per run (metrics, latency, memory, config, versions, command), summary.csv, class_size_ap.csv (4 decimals), class_size_ap_full.csv (unrounded), input_shapes.json, claims.json, params/
figures/        fig1-3 PNGs; failures/ (all candidate failure figures); report/ (the 6 used in the report, SELECTION.md)
docs/           references.md, claims_ledger.md
report/         report PDF and its HTML source
COMMANDS.md     every command in the order run
```

## Reproduce

See `COMMANDS.md`. Each `results/*.json` has the exact command in its `command` field.

## Known limitations

- The KITTI val split is the Ultralytics repackaging: DontCare regions are dropped and there are no truncation or occlusion flags, so some "false positives" may be real unlabeled objects.
- Only 223 truck instances in KITTI val, so truck AP is noisy.
- Van is merged into car; cyclist and sitting person are ignore regions; tram and misc are dropped.
- Object-size buckets differ across datasets: COCO area comes from segmentation masks, the converted KITTI file uses box area.
- No confidence intervals were computed, so small AP differences between models are not established.
- A single score cutoff (0.4) in the failure analysis is not the same operating point for every model, because scores are calibrated differently.
- No training was done, so there is no train/val leakage concern for the KITTI val split.
