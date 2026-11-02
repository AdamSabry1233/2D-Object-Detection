"""
Convert the Ultralytics-packaged KITTI val split (YOLO-format labels) into a
COCO-format JSON, using COCO category ids so that COCO-pretrained detectors'
predictions can be scored with pycocotools exactly as on COCO.
"""
import json
from collections import Counter
from pathlib import Path
from PIL import Image

KITTI_ROOT = Path("C:/dev/datasets/kitti")   # forward slashes work with Windows Python
OUT = Path(__file__).resolve().parent.parent / "annotations" / "kitti_val_coco.json"

# KITTI class id -> (COCO category id, iscrowd/ignore flag)
KITTI_TO_COCO = {
    0: (3, 0),  # car            -> car
    1: (3, 0),  # van            -> car
    2: (8, 0),  # truck          -> truck
    3: (1, 0),  # pedestrian     -> person
    4: (1, 1),  # person_sitting -> person (ignore region)
    5: (1, 1),  # cyclist        -> person (ignore region)
    # 6 tram, 7 misc -> dropped
}
KITTI_NAMES = ["car", "van", "truck", "pedestrian", "person_sitting",
               "cyclist", "tram", "misc"]
CATEGORIES = [{"id": 1, "name": "person"}, {"id": 3, "name": "car"},
              {"id": 8, "name": "truck"}]


def main():
    img_dir = KITTI_ROOT / "images" / "val"
    lbl_dir = KITTI_ROOT / "labels" / "val"
    images, annotations = [], []
    raw_counts, sizes, scale_counts = Counter(), Counter(), Counter()
    ann_id = 1

    for img_path in sorted(img_dir.glob("*.png")):
        img_id = int(img_path.stem)
        with Image.open(img_path) as im:
            W, H = im.size
        sizes[(W, H)] += 1
        images.append({"id": img_id, "file_name": img_path.name,
                       "width": W, "height": H})

        lbl_path = lbl_dir / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        for line in lbl_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls = int(parts[0])
            xc, yc, w, h = map(float, parts[1:])
            raw_counts[KITTI_NAMES[cls]] += 1
            if cls not in KITTI_TO_COCO:
                continue
            cat, ignore = KITTI_TO_COCO[cls]
            bw, bh = w * W, h * H                       # normalized -> pixels
            x, y = (xc - w / 2) * W, (yc - h / 2) * H   # center -> top-left
            area = bw * bh
            annotations.append({
                "id": ann_id, "image_id": img_id, "category_id": cat,
                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                "area": round(area, 2), "iscrowd": ignore,
            })
            ann_id += 1
            if not ignore:
                scale = "small" if area < 32**2 else "medium" if area < 96**2 else "large"
                scale_counts[scale] += 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"images": images, "annotations": annotations,
                               "categories": CATEGORIES}))

    print(f"Wrote {OUT}")
    print(f"Images: {len(images)}   Annotations kept: {len(annotations)}")
    print("Raw KITTI object counts:", dict(raw_counts))
    print("Image sizes (W,H):", dict(sizes))
    print("Scored objects by size (COCO definition):", dict(scale_counts))

    from pycocotools.coco import COCO   # sanity check: the file must parse
    COCO(str(OUT))


if __name__ == "__main__":
    main()