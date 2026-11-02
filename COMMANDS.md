# Commands, in the order run

Run from the repo root in Git Bash with the venv active (`source /c/dev/det/Scripts/activate`).

The commands in sections 3 and 4 are the exact ones recorded in the `command` field of each `results/*.json`. For sections 1 and 2 (data download and conversion) I did not log the exact commands I typed; they are reconstructed from the data sources and scripts and should be treated as equivalent, not verbatim.

## 1. Data (reconstructed)

```bash
mkdir -p /c/dev/datasets/coco /c/dev/datasets/kitti
# COCO val2017 images + annotations (official site: https://cocodataset.org/#download)
curl -L -o /c/dev/datasets/coco/val2017.zip http://images.cocodataset.org/zips/val2017.zip
curl -L -o /c/dev/datasets/coco/annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# KITTI (Ultralytics repackaging; URL from kitti.yaml)
curl -L -o /c/dev/datasets/kitti.zip https://github.com/ultralytics/assets/releases/download/v0.0.0/kitti.zip
# unzip each archive so that these exist:
#   C:/dev/datasets/coco/val2017/                     C:/dev/datasets/coco/annotations/instances_val2017.json
#   C:/dev/datasets/kitti/images/val/                 C:/dev/datasets/kitti/labels/val/
```

## 2. KITTI to COCO conversion

```bash
python scripts/convert_kitti_to_coco.py        # writes annotations/kitti_val_coco.json
```

## 3. Smoke test (recorded; not used in any reported result)

```bash
python scripts/eval_detector.py --model yolo11s --dataset coco --limit 50
```

## 4. The six core evaluations (recorded)

```bash
python scripts/eval_detector.py --model yolo11s --dataset coco
python scripts/eval_detector.py --model yolo11s --dataset kitti
python scripts/eval_detector.py --model rtdetr-l --dataset coco
python scripts/eval_detector.py --model rtdetr-l --dataset kitti
python scripts/eval_detector.py --model frcnn --dataset coco
python scripts/eval_detector.py --model frcnn --dataset kitti
```

Each writes `results/<model>_<dataset>.json` and raw detections to `C:/dev/runs/<model>_<dataset>_dets.json`.

## 5. Resolution experiment (recorded)

```bash
python scripts/eval_detector.py --model yolo11s --dataset kitti --imgsz 1280
```

Writes `results/yolo11s_kitti_imgsz1280.json`. Its latency is not comparable to the 640 runs.

## 6. Analysis and figures

```bash
python scripts/count_params.py                 # results/params/model_params.json
python scripts/summarize.py                    # results/summary.csv
python scripts/class_size_breakdown.py         # results/class_size_ap.csv, results/class_size_ap_full.csv (unrounded)  (printed table: results/class_size_table.txt)
python scripts/failure_analysis.py             # figures/failures/*  (defaults: score >= 0.4, IoU >= 0.5; printed table: figures/failures/miss_rates.txt)
python scripts/measure_input_shapes.py         # results/input_shapes.json (real network input shape per model on one KITTI image)
python scripts/make_plots.py                   # figures/fig1-3
python scripts/make_report_failures.py         # figures/report/compact/* (report-sized versions of the 6 chosen failure figures in figures/report/)
```

## 7. Report

```bash
python scripts/claims_check.py                 # computes every derived number; used by the ledger and the report
python scripts/build_report.py                 # report/report.html and report/CMPE249_HW_2D_Detection_TrackA.pdf
```
