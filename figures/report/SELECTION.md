# Failure-analysis figure selection

Files in this folder are the original figures from `figures/failures/`. The report embeds the report-sized versions in `compact/` (made by `scripts/make_report_failures.py`): same images and same matching, with each panel cropped vertically to the band that contains the boxes and class/score labels drawn only on large boxes. The observations below were made on the full-size originals.

All panels: KITTI val, score >= 0.4, IoU >= 0.5. Top to bottom: YOLO11s, RT-DETR-L, Faster R-CNN.
Colors: blue = correct detection, red = false positive (score shown), orange = missed ground truth, dashed gray = ignore region.
Counts (missed / false positives) come from `figures/failures/index.csv`; "correct" counts come from the panel titles in the images.
"WHAT IS VISIBLE" lists only what I could see in the image. "Likely cause" is hedged and is a hypothesis, not a tested result.

## missed_by_all

### 1. `missed_by_all_1_005226.png` (KITTI 005226)
- Counts: missed y11s 11 / rtdetr 10 / frcnn 11; false positives 3 / 7 / 13. Correct: 5 / 6 / 5.
- WHAT IS VISIBLE: At the left edge, a large dark-blue vehicle cut off by the image border. The ground truth there is an orange "car" box. YOLO11s and RT-DETR-L each draw one red "truck" box (0.87 and 0.78) over it; Faster R-CNN draws two red "truck" boxes (0.71 and 0.41). In the center is a crowd of small people standing among bicycles, mostly orange (missed) boxes. The two women walking at the right are detected (blue) by all three models.
- Likely cause: consistent with (a) a class-label disagreement on the large vehicle, which looks van- or truck-like to me, so the models call it truck while the label is car, and (b) small, closely spaced, partly overlapping pedestrians. I cannot tell from the image what the vehicle's original KITTI class was.
- Measured link: person miss rate at score >= 0.4 is 0.489 / 0.403 / 0.347 (YOLO11s / RT-DETR-L / Faster R-CNN), and small KITTI person AP is about 0.02-0.04 for all three models (`results/class_size_ap.csv`).
- Red boxes: the "truck" boxes look like a class error against the label, not an unlabeled object. The few small red person boxes in the crowd: cannot tell.

### 2. `missed_by_all_4_005245.png` (KITTI 005245)
- Counts: missed 10 / 10 / 9; false positives 2 / 7 / 15. Correct: 6 / 6 / 7.
- WHAT IS VISIBLE: A narrow street with parked cars along the left, several overlapping orange boxes (car, truck, person) behind and around the near car, and a car cut off by the right image border (orange, missed by all). An elderly person walking on the right is detected (blue). Faster R-CNN draws many small red car boxes on the right side, near a row of cars far down the street. YOLO11s draws a red "car 0.42" box over a dark shape left of the walking person.
- Likely cause: consistent with occlusion and heavy overlap between parked cars, plus truncation at the image border. Heavy overlap between objects is also where NMS-based suppression could matter, but I did not test that.
- Red boxes: the small red boxes on the right look like real cars that KITTI did not label (distant), but I cannot be sure at this size. The dark shape with the YOLO11s "car 0.42" box: cannot tell what it is.

## disagreement

### 3. `disagreement_1_003554.png` (KITTI 003554)
- Counts: missed 13 / 4 / 6; false positives 0 / 2 / 14. Correct: 1 / 10 / 8.
- WHAT IS VISIBLE: A wide road with a row of distant cars across the middle of the frame. YOLO11s detects 1 and misses 13 (almost all orange). RT-DETR-L detects 10 with 2 false positives. Faster R-CNN detects 8 with 14 false positives, including a large red "truck 0.53" box over the billboard area and several red boxes at the right edge.
- Likely cause: consistent with small, distant objects being the hard case for the 640-px YOLO11s input (the long side of KITTI shrinks to 640, scale about 0.52). This fits the measured small-object miss rate for YOLO11s (0.554) vs. RT-DETR-L (0.297) and Faster R-CNN (0.223). I did not test resolution in this image; the separate 1280 experiment raises YOLO11s KITTI AP only from 0.241 to 0.262.
- Red boxes: the Faster R-CNN "truck" box over the billboard looks like a model error. The clustered small red boxes on the right could be real unlabeled cars: cannot tell.

### 4. `disagreement_2_003219.png` (KITTI 003219)
- Counts: missed 10 / 8 / 4; false positives 3 / 2 / 19. Correct: 10 / 12 / 16.
- WHAT IS VISIBLE: A street lined with parked cars; a large red car at the front right (detected by all). Faster R-CNN has the most correct detections (16) and the fewest misses (4), but also 19 false positives. Most of these are a dense stack of small red car boxes down the far end of the road.
- Likely cause: consistent with Faster R-CNN's larger working resolution on KITTI (about 1333 x 402, a scale of about 1.07 vs. about 0.52 for the 640-px models) helping it find small distant cars at the cost of more low-quality extra boxes. This is a hypothesis, not tested.
- Red boxes: the many small red boxes at the far end look like real distant cars that KITTI did not label, but I cannot confirm that.

## false_positives

### 5. `false_positives_2_000018.png` (KITTI 000018)
- Counts: missed 1 / 1 / 0; false positives 10 / 11 / 26. Correct: 1 / 1 / 2.
- WHAT IS VISIBLE: An almost empty road. KITTI labels only 2 cars. The red car boxes (score 0.41 to 0.81) sit on vehicles that appear to be parked along the grass strip on the right and the road edge on the left. Faster R-CNN has 26 red boxes, many at scores near 0.9.
- Likely cause: consistent with real but unlabeled objects. These cars look real to me, but they are far away, so KITTI may have left them unlabeled. If so, these are not model errors and they lower the measured precision (and AP) of all models. Faster R-CNN's very high scores on them (about 0.9+) point the same way.
- Red boxes: I judge most red car boxes to be real but unlabeled objects; I cannot be certain at this size.

### 6. `false_positives_3_006255.png` (KITTI 006255)
- Counts: missed 1 / 1 / 1; false positives 8 / 10 / 28. Correct: 1 / 1 / 1.
- WHAT IS VISIBLE: The same kind of road scene as figure 5. YOLO11s has a red "person" box on a tall thin object at the left roadside. Faster R-CNN has two red "person" boxes on the left side, as well as many red car boxes on the right.
- Likely cause: the red person boxes are consistent with a roadside post or bollard being mistaken for a person; I cannot tell what the objects are at this size. The red car boxes on the right are consistent with unlabeled parked cars, as in figure 5.
- Red boxes: red car boxes - probably real but unlabeled. Red person boxes - possibly model errors, but I cannot tell.
