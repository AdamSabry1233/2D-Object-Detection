# Verified references

Each entry was checked on 2026-11-01 against the source named in "Verified via". Candidates I could not verify were dropped (see the end).

## Papers

1. T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollar, C. L. Zitnick, "Microsoft COCO: Common Objects in Context," ECCV 2014, pp. 740-755 (Springer).
   URL: https://arxiv.org/abs/1405.0312 (DOI 10.1007/978-3-319-10602-1_48).
   Verified via: arXiv page (title, authors; the arXiv author list is longer than the ECCV list, so the report cites "Lin et al."); venue and pages from search results (Springer/ECCV 2014 listings). The Springer page itself required a login redirect and was not opened.
2. A. Geiger, P. Lenz, R. Urtasun, "Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite," CVPR 2012, pp. 3354-3361.
   URL: https://www.cvlibs.net/publications/Geiger2012CVPR.pdf (DOI 10.1109/CVPR.2012.6248074).
   Verified via: first page of the PDF (title, authors); venue and pages from search results.
3. A. Geiger, P. Lenz, C. Stiller, R. Urtasun, "Vision meets Robotics: The KITTI Dataset," International Journal of Robotics Research (IJRR), vol. 32, no. 11, 2013.
   URL: https://www.cvlibs.net/publications/Geiger2013IJRR.pdf (DOI 10.1177/0278364913491297).
   Verified via: first page of the PDF (title, authors); journal, volume and DOI from search results (the page range differs between sources, so the report omits it).
4. S. Ren, K. He, R. Girshick, J. Sun, "Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks," NeurIPS 2015 (Advances in Neural Information Processing Systems 28), pp. 91-99.
   URL: https://arxiv.org/abs/1506.01497 ; https://papers.nips.cc/paper/2015/hash/14bfa6bb14875e45bba028a21ed38046-Abstract.html
   Verified via: arXiv page (title, authors) and the NeurIPS proceedings listing returned by search.
5. T.-Y. Lin, P. Dollar, R. Girshick, K. He, B. Hariharan, S. Belongie, "Feature Pyramid Networks for Object Detection," CVPR 2017, pp. 2117-2125.
   URL: https://arxiv.org/abs/1612.03144 ; https://openaccess.thecvf.com/content_cvpr_2017/html/Lin_Feature_Pyramid_Networks_CVPR_2017_paper.html
   Verified via: arXiv page (title, authors); CVPR 2017 open-access listing as returned by search (the CVF site itself returned HTTP 403 to my fetch).
6. K. He, X. Zhang, S. Ren, J. Sun, "Deep Residual Learning for Image Recognition," CVPR 2016, pp. 770-778.
   URL: https://arxiv.org/abs/1512.03385 ; https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html
   Verified via: arXiv page; CVPR 2016 open-access listing as returned by search (CVF fetch returned 403).
7. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, S. Zagoruyko, "End-to-End Object Detection with Transformers," ECCV 2020.
   URL: https://arxiv.org/abs/2005.12872 ; https://www.ecva.net/papers/eccv_2020/papers_ECCV/html/832_ECCV_2020_paper.php
   Verified via: arXiv page; ECVA ECCV 2020 listing as returned by search.
8. Y. Zhao, W. Lv, S. Xu, J. Wei, G. Wang, Q. Dang, Y. Liu, J. Chen, "DETRs Beat YOLOs on Real-time Object Detection," CVPR 2024, pp. 16965-16974.
   URL: https://arxiv.org/abs/2304.08069 ; https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html
   Verified via: arXiv page; CVPR 2024 open-access listing as returned by search (CVF fetch returned 403).
9. J. Redmon, S. Divvala, R. Girshick, A. Farhadi, "You Only Look Once: Unified, Real-Time Object Detection," CVPR 2016, pp. 779-788.
   URL: https://arxiv.org/abs/1506.02640 ; https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html
   Verified via: arXiv page; CVPR 2016 open-access listing as returned by search (CVF fetch returned 403).

## Official docs, repositories and benchmark pages

10. Ultralytics, "YOLO11" model documentation. https://docs.ultralytics.com/models/yolo11/ (fetched; the detection table lists YOLO11s at 47.0 mAP val 50-95, imgsz 640, "Detection (COCO)"). YOLO11 has no paper, so this page is the citation.
11. Ultralytics, "RT-DETR" model documentation. https://docs.ultralytics.com/models/rtdetr/ (fetched; states "RT-DETR-L: 53.0% AP on COCO val2017").
12. Ultralytics repository. https://github.com/ultralytics/ultralytics (fetched; AGPL-3.0; lists YOLO11).
13. PyTorch, torchvision detection models documentation, `fasterrcnn_resnet50_fpn`. https://docs.pytorch.org/vision/stable/models/faster_rcnn.html (fetched; describes the model as the Faster R-CNN ResNet-50-FPN model; the published box mAP is read from the installed torchvision weights metadata, see below).
14. COCO API repository (pycocotools source). https://github.com/cocodataset/cocoapi (fetched; belongs to the `cocodataset` organization).
15. KITTI Vision Benchmark Suite, 2D object detection benchmark. https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d (fetched; confirms the official benchmark page, which names cars, pedestrians and cyclists).

## Published COCO val mAP for the evaluated checkpoints

Stored machine-readably in `docs/published_map.json`.

| Model | Published mAP@[.5:.95] | Source |
|---|---|---|
| YOLO11s | 47.0 (imgsz 640) | [10]. The page labels the table "Detection (COCO)" and does not name the split. |
| RT-DETR-L | 53.0 | [11]. States COCO val2017. |
| Faster R-CNN R50-FPN, COCO_V1 | 37.0 | torchvision `FasterRCNN_ResNet50_FPN_Weights.COCO_V1.meta["_metrics"]["COCO-val2017"]["box_map"]`, read from the installed library (torchvision 0.21.0); parameter count 41,755,286 in the same metadata matches `results/params/model_params.json`. The docs page itself did not render the figure to my fetch. |

## Dropped (could not verify)

- COCO detection evaluation page (https://cocodataset.org/#detection-eval): the fetch returned only navigation text, so I could not confirm the metric definitions on it. The COCO API repository [14] is cited instead.
