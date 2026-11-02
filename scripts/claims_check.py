"""
Compute every number the report uses from files in this repo and write
  results/claims.json      (machine-readable values, read by build_report.py)
  docs/claims_ledger.md    (claim | source file | value | status)

Statuses: MEASURED (read directly from a result file), DERIVED (computed from
result files), HYPOTHESIS (explanation not isolated by an experiment),
UNVERIFIED (observed but not explained or not checkable here).
"""
import csv, json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
MODELS = ["yolo11s", "rtdetr-l", "frcnn"]
LABEL = {"yolo11s": "YOLO11s", "rtdetr-l": "RT-DETR-L", "frcnn": "Faster R-CNN"}
CLASSES = ["person", "car", "truck"]
SIZES = ["small", "medium", "large"]
MIN_N = 50          # cells with fewer objects than this are flagged unreliable
M3 = "3class_person_car_truck"


def pct(new, old):
    return (new - old) / old * 100.0


def load():
    R = {(m, ds): json.loads((RES / f"{m}_{ds}.json").read_text())
         for m in MODELS for ds in ("coco", "kitti")}
    R1280 = json.loads((RES / "yolo11s_kitti_imgsz1280.json").read_text())
    params = json.loads((RES / "params" / "model_params.json").read_text())
    published = json.loads((ROOT / "docs" / "published_map.json").read_text())
    with open(RES / "class_size_ap_full.csv") as f:
        cs = {(r["dataset"], r["model"], r["cls"]): r for r in csv.DictReader(f)}
    with open(ROOT / "figures" / "failures" / "miss_rates.csv") as f:
        miss = {r["model"]: r for r in csv.DictReader(f)}
    kgt = json.loads((ROOT / "annotations" / "kitti_val_coco.json").read_text())
    return R, R1280, params, published, cs, miss, kgt


def compute():
    R, R1280, params, published, cs, miss, kgt = load()
    claims, D = [], {}

    def add(cid, text, source, value, status):
        claims.append(dict(id=cid, claim=text, source=source, value=value, status=status))

    def ap3(m, ds, key="AP"):
        return R[(m, ds)]["metrics"][M3]["stats"][key]

    def pc(m, ds, c):
        return R[(m, ds)]["metrics"][M3]["per_class_AP"][c]

    # ---- dataset facts ----
    n_img = {ds: R[("yolo11s", ds)]["n_images"] for ds in ("coco", "kitti")}
    D["n_images"] = n_img
    add("n_images", "Images evaluated (COCO val2017 / KITTI val)", "results/*_coco.json, *_kitti.json",
        f"{n_img['coco']} / {n_img['kitti']}", "MEASURED")
    nk = Counter(a["category_id"] for a in kgt["annotations"] if not a["iscrowd"])
    n_ign = sum(1 for a in kgt["annotations"] if a["iscrowd"])
    D["kitti_counts"] = {"person": nk[1], "car": nk[3], "truck": nk[8], "ignore": n_ign}
    add("kitti_counts", "KITTI val objects after mapping (person / car / truck / ignore regions)",
        "annotations/kitti_val_coco.json", f"{nk[1]} / {nk[3]} / {nk[8]} / {n_ign}", "MEASURED")

    # size distributions (non-ignore person/car/truck), from per-cell counts
    dist = {}
    for ds in ("coco", "kitti"):
        tot = {s: sum(int(cs[(ds, "frcnn", c)][f"n_{s}"]) for c in CLASSES) for s in SIZES}
        n = sum(tot.values())
        dist[ds] = {"n": n, **{s: tot[s] / n * 100 for s in SIZES}}
        add(f"size_dist_{ds}", f"{ds.upper()} person/car/truck size distribution small/medium/large (n={n})",
            "results/class_size_ap.csv",
            " / ".join(f"{dist[ds][s]:.0f}%" for s in SIZES), "DERIVED")
    D["size_dist"] = dist
    car_small = {ds: int(cs[(ds, "frcnn", "car")]["n_small"]) /
                 sum(int(cs[(ds, "frcnn", "car")][f"n_{s}"]) for s in SIZES) * 100
                 for ds in ("coco", "kitti")}
    D["car_small_share"] = car_small
    add("car_small_share", "Share of cars that are small: COCO vs KITTI", "results/class_size_ap.csv",
        f"{car_small['coco']:.0f}% vs {car_small['kitti']:.0f}%", "DERIVED")

    # ---- headline 3-class AP and drops ----
    D["ap3"] = {f"{m}_{ds}": ap3(m, ds) for m in MODELS for ds in ("coco", "kitti")}
    D["drop_overall"] = {m: pct(ap3(m, "kitti"), ap3(m, "coco")) for m in MODELS}
    for m in MODELS:
        add(f"drop_overall_{m}", f"{LABEL[m]} 3-class AP, COCO -> KITTI",
            f"results/{m}_coco.json, {m}_kitti.json",
            f"{ap3(m,'coco'):.3f} -> {ap3(m,'kitti'):.3f} ({D['drop_overall'][m]:+.1f}%)", "DERIVED")
    rank = {ds: sorted(MODELS, key=lambda m: -ap3(m, ds)) for ds in ("coco", "kitti")}
    D["rank"] = rank
    for ds in ("coco", "kitti"):
        add(f"rank_{ds}", f"Ranking by 3-class AP on {ds.upper()}", "results/*.json",
            " > ".join(LABEL[m] for m in rank[ds]), "DERIVED")
    gap = ap3("rtdetr-l", "kitti") - ap3("frcnn", "kitti")
    D["gap_rtdetr_frcnn_kitti"] = gap
    add("gap_rtdetr_frcnn_kitti", "RT-DETR-L minus Faster R-CNN, KITTI 3-class AP (no confidence intervals computed)",
        "results/rtdetr-l_kitti.json, frcnn_kitti.json",
        f"{ap3('rtdetr-l','kitti'):.3f} vs {ap3('frcnn','kitti'):.3f} (diff {gap:+.3f})", "DERIVED")

    # ---- per class ----
    D["per_class"] = {f"{m}_{ds}_{c}": pc(m, ds, c) for m in MODELS for ds in ("coco", "kitti") for c in CLASSES}
    D["drop_class"] = {}
    for m in MODELS:
        for c in CLASSES:
            d = pct(pc(m, "kitti", c), pc(m, "coco", c))
            D["drop_class"][f"{m}_{c}"] = d
            add(f"drop_class_{m}_{c}", f"{LABEL[m]} {c} AP, COCO -> KITTI",
                f"results/{m}_coco.json, {m}_kitti.json",
                f"{pc(m,'coco',c):.3f} -> {pc(m,'kitti',c):.3f} ({d:+.1f}%)", "DERIVED")

    # ---- per size (pooled) ----
    D["drop_size"] = {}
    for m in MODELS:
        for k, nm in (("AP_small", "small"), ("AP_medium", "medium"), ("AP_large", "large")):
            d = pct(ap3(m, "kitti", k), ap3(m, "coco", k))
            D["drop_size"][f"{m}_{nm}"] = d
            add(f"drop_size_{m}_{nm}", f"{LABEL[m]} pooled {nm}-object AP, COCO -> KITTI",
                f"results/{m}_coco.json, {m}_kitti.json",
                f"{ap3(m,'coco',k):.3f} -> {ap3(m,'kitti',k):.3f} ({d:+.1f}%)", "DERIVED")

    # ---- class x size ----
    cells = []
    for m in MODELS:
        for c in CLASSES:
            for s in SIZES:
                rc, rk = cs[("coco", m, c)], cs[("kitti", m, c)]
                a_c, a_k = float(rc[f"AP_{s}"]), float(rk[f"AP_{s}"])
                n_c, n_k = int(rc[f"n_{s}"]), int(rk[f"n_{s}"])
                cells.append(dict(model=m, cls=c, size=s, ap_coco=a_c, ap_kitti=a_k,
                                  drop_pct=pct(a_k, a_c), n_coco=n_c, n_kitti=n_k,
                                  reliable=(n_c >= MIN_N and n_k >= MIN_N)))
    D["cells"] = cells
    unrel = [x for x in cells if not x["reliable"]]
    add("cells_unreliable", f"class x size cells with fewer than {MIN_N} objects in COCO or KITTI (flagged unreliable)",
        "results/class_size_ap.csv",
        "; ".join(sorted({f"{x['cls']}/{x['size']} (n_coco={x['n_coco']}, n_kitti={x['n_kitti']})" for x in unrel})),
        "DERIVED")
    car_cells = [x for x in cells if x["cls"] == "car" and x["reliable"]]
    n_drop = sum(1 for x in car_cells if x["drop_pct"] < 0)
    D["car_cells_dropping"] = [n_drop, len(car_cells)]
    add("car_cells_dropping", "Reliable car class-x-size cells (3 models x 3 sizes) where AP is lower on KITTI than COCO",
        "results/class_size_ap.csv", f"{n_drop} of {len(car_cells)}", "DERIVED")
    for x in car_cells:
        if x["model"] == "frcnn" and x["size"] == "large":
            add("frcnn_car_large", "Faster R-CNN large-car AP, COCO vs KITTI", "results/class_size_ap.csv",
                f"{x['ap_coco']:.3f} vs {x['ap_kitti']:.3f} ({x['drop_pct']:+.1f}%)", "DERIVED")
    # the pooled-small-object distortion
    fs = cs[("kitti", "frcnn", "truck")]
    add("frcnn_small_truck", "Faster R-CNN KITTI truck AP small / medium / large and n small",
        "results/class_size_ap.csv",
        f"{float(fs['AP_small']):.3f} / {float(fs['AP_medium']):.3f} / {float(fs['AP_large']):.3f}; n_small={fs['n_small']}",
        "MEASURED")
    D["frcnn_kitti_truck"] = {s: float(fs[f"AP_{s}"]) for s in SIZES}
    D["frcnn_kitti_truck"]["n_small"] = int(fs["n_small"])
    sp = {m: float(cs[("kitti", m, "person")]["AP_small"]) for m in MODELS}
    D["kitti_small_person_ap"] = sp
    add("kitti_small_person", "KITTI small-person AP (YOLO11s / RT-DETR-L / Faster R-CNN)", "results/class_size_ap.csv",
        " / ".join(f"{sp[m]:.3f}" for m in MODELS), "MEASURED")

    # ---- AP50 / AP75 ----
    D["ap50_ap75"] = {f"{m}_{ds}": [ap3(m, ds, "AP50"), ap3(m, ds, "AP75")] for m in MODELS for ds in ("coco", "kitti")}
    D["ar100"] = {f"{m}_{ds}": ap3(m, ds, "AR100") for m in MODELS for ds in ("coco", "kitti")}
    D["ap75_over_ap50"] = {}
    for m in MODELS:
        rc = ap3(m, "coco", "AP75") / ap3(m, "coco", "AP50")
        rk = ap3(m, "kitti", "AP75") / ap3(m, "kitti", "AP50")
        D["ap75_over_ap50"][m] = [rc, rk]
        add(f"ap75_ap50_{m}", f"{LABEL[m]} AP75/AP50 ratio on COCO vs KITTI (3-class; lower = relatively looser boxes)",
            f"results/{m}_coco.json, {m}_kitti.json", f"{rc:.2f} vs {rk:.2f}", "DERIVED")
    D["person_drop_size"] = {}
    for m in MODELS:
        for s in SIZES:
            x = next(c for c in cells if c["model"] == m and c["cls"] == "person" and c["size"] == s)
            D["person_drop_size"][f"{m}_{s}"] = x["drop_pct"]
            add(f"person_drop_{m}_{s}", f"{LABEL[m]} {s}-person AP, COCO -> KITTI (n_coco={x['n_coco']}, n_kitti={x['n_kitti']})",
                "results/class_size_ap.csv", f"{x['ap_coco']:.3f} -> {x['ap_kitti']:.3f} ({x['drop_pct']:+.1f}%)", "DERIVED")
    fc = json.loads((ROOT / "figures" / "report" / "counts.json").read_text())
    D["fail_counts"] = fc
    add("failure_counts", "Correct / false-positive / missed counts per model for the six selected failure images (score>=0.4, IoU>=0.5)",
        "figures/report/counts.json (scripts/make_report_failures.py)", f"{len(fc)} images x 3 models", "MEASURED")
    add("table_headline", "AP, AP50, AP75, AP small/medium/large, AR100 and per-class AP for every model and dataset (results tables)",
        "results/<model>_<dataset>.json (also results/summary.csv)", "read directly", "MEASURED")
    add("table_class_size", "Per class x size AP and object counts for every model and dataset (class x size table)",
        "results/class_size_ap.csv", "read directly", "MEASURED")
    add("table_latency", "Mean/median/p95 latency, FPS and peak GPU memory for every model and dataset",
        "results/<model>_<dataset>.json", "read directly", "MEASURED")
    cfg = R[("yolo11s", "coco")]["config"]
    add("config", "Protocol: confidence threshold, max detections, NMS IoU per model, hardware and software versions",
        "results/<model>_<dataset>.json (config, hardware, versions)",
        f"conf={cfg['conf_thresh']}, max_det={cfg['max_det']}; {R[('yolo11s','coco')]['hardware']}; "
        f"torch {R[('yolo11s','coco')]['versions']['torch']}", "MEASURED")

    # ---- 80-class sanity check vs published ----
    D["ap80"] = {}
    for m in MODELS:
        got = R[(m, "coco")]["metrics"]["coco_80class"]["stats"]["AP"]
        pub = published[m]["mAP_50_95"] / 100.0
        D["ap80"][m] = dict(measured=got, published=pub, diff=got - pub)
        add(f"ap80_{m}", f"{LABEL[m]} 80-class COCO AP: measured vs published", f"results/{m}_coco.json, docs/published_map.json",
            f"{got:.3f} vs {pub:.3f} (diff {got-pub:+.3f})", "DERIVED")

    # ---- input size actually seen by each model (measured by scripts/measure_input_shapes.py) ----
    sizes = Counter((i["width"], i["height"]) for i in kgt["images"])
    (w, h), cnt = sizes.most_common(1)[0]
    D["resolution"] = dict(w=w, h=h, count=cnt)
    sh = json.loads((RES / "input_shapes.json").read_text())
    ow, oh = sh["orig_w"], sh["orig_h"]
    yo, y12, rt, fr = sh["yolo11s_640"], sh["yolo11s_1280"], sh["rtdetr-l_640"], sh["frcnn"]
    D["inputs"] = {
        "yolo11s_640": dict(h=yo["input_h"], w=yo["input_w"], px=yo["input_h"] * yo["input_w"],
                            sx=yo["input_w"] / ow, sy=yo["input_w"] / ow),
        "yolo11s_1280": dict(h=y12["input_h"], w=y12["input_w"], px=y12["input_h"] * y12["input_w"],
                             sx=y12["input_w"] / ow, sy=y12["input_w"] / ow),
        "rtdetr-l_640": dict(h=rt["input_h"], w=rt["input_w"], px=rt["input_h"] * rt["input_w"],
                             sx=rt["input_w"] / ow, sy=rt["input_h"] / oh),
        "frcnn": dict(h=fr["padded_h"], w=fr["padded_w"], px=fr["padded_h"] * fr["padded_w"],
                      sx=fr["resized_w"] / ow, sy=fr["resized_h"] / oh,
                      resized_h=fr["resized_h"], resized_w=fr["resized_w"]),
        "orig": dict(w=ow, h=oh, image=sh["image"]),
        "rtdetr_scale_fill": rt["scale_fill"], "yolo_scale_fill": yo["scale_fill"],
    }
    I = D["inputs"]
    px = {k: I[k]["px"] for k in ("yolo11s_640", "yolo11s_1280", "rtdetr-l_640", "frcnn")}
    D["input_px"] = dict(px, yolo1280_over_rtdetr=px["yolo11s_1280"] / px["rtdetr-l_640"],
                         yolo640_over_rtdetr=px["yolo11s_640"] / px["rtdetr-l_640"],
                         yolo1280_over_frcnn=px["yolo11s_1280"] / px["frcnn"],
                         diff_1280_vs_rtdetr=abs(px["yolo11s_1280"] - px["rtdetr-l_640"]),
                         diff_640_vs_rtdetr=abs(px["yolo11s_640"] - px["rtdetr-l_640"]))
    D["input_px"]["1280_closer_to_rtdetr_than_640"] = D["input_px"]["diff_1280_vs_rtdetr"] < D["input_px"]["diff_640_vs_rtdetr"]
    add("input_shapes", f"Network input per model for one {ow}x{oh} KITTI image ({sh['image']}): YOLO11s 640 / YOLO11s 1280 / RT-DETR-L / Faster R-CNN (height x width; Faster R-CNN resized size, tensor padded)",
        "results/input_shapes.json (scripts/measure_input_shapes.py); ultralytics/models/rtdetr/predict.py scale_fill",
        f"{yo['input_h']}x{yo['input_w']} / {y12['input_h']}x{y12['input_w']} / {rt['input_h']}x{rt['input_w']} (stretched, scale_fill={rt['scale_fill']}) / "
        f"{fr['resized_h']}x{fr['resized_w']} (tensor {fr['padded_h']}x{fr['padded_w']})", "MEASURED")
    add("input_scales", "Scale factor (horizontal / vertical) from the original image: YOLO11s 640 / YOLO11s 1280 / RT-DETR-L / Faster R-CNN",
        "results/input_shapes.json", " / ".join(f"{I[k]['sx']:.2f}/{I[k]['sy']:.2f}" for k in ("yolo11s_640", "yolo11s_1280", "rtdetr-l_640", "frcnn")), "DERIVED")
    add("input_pixels", "Network input pixels: YOLO11s 640 / YOLO11s 1280 / RT-DETR-L / Faster R-CNN (padded tensors)",
        "results/input_shapes.json", " / ".join(f"{px[k]:,}" for k in ("yolo11s_640", "yolo11s_1280", "rtdetr-l_640", "frcnn")), "DERIVED")
    add("input_px_1280", "YOLO11s 1280 input pixels relative to RT-DETR-L 640 (and 640-run relative); absolute differences; 1280 closer than 640",
        "results/input_shapes.json",
        f"{D['input_px']['yolo1280_over_rtdetr']:.2f}x (640: {D['input_px']['yolo640_over_rtdetr']:.2f}x); "
        f"{D['input_px']['diff_1280_vs_rtdetr']:,} vs {D['input_px']['diff_640_vs_rtdetr']:,}; {D['input_px']['1280_closer_to_rtdetr_than_640']}", "DERIVED")
    add("input_px_1280_frcnn", "YOLO11s 1280 input pixels relative to Faster R-CNN", "results/input_shapes.json",
        f"{D['input_px']['yolo1280_over_frcnn']:.2f}x", "DERIVED")
    a640, a1280 = ap3("yolo11s", "kitti"), R1280["metrics"][M3]["stats"]["AP"]
    D["res1280"] = dict(ap_640=a640, ap_1280=a1280, delta=a1280 - a640, delta_pct=pct(a1280, a640),
                        ap50=R1280["metrics"][M3]["stats"]["AP50"], ap75=R1280["metrics"][M3]["stats"]["AP75"],
                        ap_small=R1280["metrics"][M3]["stats"]["AP_small"],
                        ap_small_640=ap3("yolo11s", "kitti", "AP_small"),
                        per_class=R1280["metrics"][M3]["per_class_AP"],
                        latency_ms=R1280["latency_ms"]["mean"], fps=R1280["fps"], mem=R1280["peak_gpu_mem_gb"],
                        gap_to_rtdetr=ap3("rtdetr-l", "kitti") - a1280, gap_to_frcnn=ap3("frcnn", "kitti") - a1280)
    pc1280 = R1280["metrics"][M3]["per_class_AP"]
    D["res1280"]["per_class_delta"] = {c: pc1280[c] - pc("yolo11s", "kitti", c) for c in CLASSES}
    D["res1280"]["per_class_640"] = {c: pc("yolo11s", "kitti", c) for c in CLASSES}
    add("res1280_per_class", "YOLO11s KITTI per-class AP at imgsz 640 -> 1280 (person / car / truck; single run)",
        "results/yolo11s_kitti.json, yolo11s_kitti_imgsz1280.json",
        "; ".join(f"{c} {pc('yolo11s','kitti',c):.3f} -> {pc1280[c]:.3f} ({pc1280[c]-pc('yolo11s','kitti',c):+.3f})" for c in CLASSES), "DERIVED")
    add("res1280", "YOLO11s KITTI 3-class AP at imgsz 640 -> 1280", "results/yolo11s_kitti.json, yolo11s_kitti_imgsz1280.json",
        f"{a640:.3f} -> {a1280:.3f} ({a1280-a640:+.3f}, {pct(a1280,a640):+.1f}%)", "DERIVED")
    add("res1280_gaps", "KITTI AP gap that remains at imgsz 1280 vs RT-DETR-L and vs Faster R-CNN",
        "results/*_kitti*.json", f"{D['res1280']['gap_to_rtdetr']:+.3f} / {D['res1280']['gap_to_frcnn']:+.3f}", "DERIVED")
    add("res1280_latency", "YOLO11s KITTI mean latency at 640 vs 1280 (not comparable, see report)",
        "results/yolo11s_kitti.json, yolo11s_kitti_imgsz1280.json",
        f"{R[('yolo11s','kitti')]['latency_ms']['mean']} ms vs {R1280['latency_ms']['mean']} ms", "UNVERIFIED")

    # ---- latency / fps / memory ----
    D["lat"] = {}
    for m in MODELS:
        for ds in ("coco", "kitti"):
            r = R[(m, ds)]
            D["lat"][f"{m}_{ds}"] = dict(mean=r["latency_ms"]["mean"], median=r["latency_ms"]["median"],
                                          p95=r["latency_ms"]["p95"], fps=r["fps"], mem=r["peak_gpu_mem_gb"])
        lc, lk = R[(m, "coco")]["latency_ms"]["mean"], R[(m, "kitti")]["latency_ms"]["mean"]
        ch = pct(lk, lc)
        D["lat"][f"{m}_change_pct"] = ch
        add(f"lat_{m}", f"{LABEL[m]} mean latency COCO -> KITTI (>20% change treated as more than noise)",
            f"results/{m}_coco.json, {m}_kitti.json",
            f"{lc} ms -> {lk} ms ({ch:+.0f}%); peak mem {R[(m,'coco')]['peak_gpu_mem_gb']} -> {R[(m,'kitti')]['peak_gpu_mem_gb']} GB",
            "DERIVED")
    D["speed"] = {}
    for ds in ("coco", "kitti"):
        ly, lr, lf = (R[(m, ds)]["latency_ms"]["mean"] for m in ("yolo11s", "rtdetr-l", "frcnn"))
        ay, ar_, af = (ap3(m, ds) for m in ("yolo11s", "rtdetr-l", "frcnn"))
        D["speed"][ds] = dict(rtdetr_over_yolo_latency=lr / ly, ap_gain_rtdetr_over_yolo=ar_ - ay,
                              ap_gain_rtdetr_over_yolo_pct=pct(ar_, ay), frcnn_over_rtdetr_latency=lf / lr,
                              ap_rtdetr_minus_frcnn=ar_ - af, frcnn_over_yolo_latency=lf / ly,
                              rtdetr_faster_and_at_least_as_accurate_as_frcnn=(lr < lf and ar_ >= af))
        add(f"speed_{ds}", f"{ds.upper()}: RT-DETR-L/YOLO11s latency ratio; RT-DETR-L AP gain over YOLO11s; "
            "Faster R-CNN/RT-DETR-L latency ratio; RT-DETR-L AP minus Faster R-CNN AP; RT-DETR-L faster and at least as accurate as Faster R-CNN",
            "results/*.json",
            f"{lr/ly:.1f}x; {ar_-ay:+.3f} ({pct(ar_, ay):+.1f}%); {lf/lr:.1f}x; {ar_-af:+.3f}; "
            f"{D['speed'][ds]['rtdetr_faster_and_at_least_as_accurate_as_frcnn']}", "DERIVED")
    add("lat_frcnn_cause", "Faster R-CNN's lower KITTI latency is due to fewer pixels after torchvision's resize",
        "(not tested)", "plausible, untested", "HYPOTHESIS")
    add("lat_yolo_unexplained", "YOLO11s' slightly higher KITTI latency", "(not tested)", "unexplained", "UNVERIFIED")
    add("lat_rtdetr_fixed", "RT-DETR-L's near-identical latency and memory on both datasets reflect its fixed 640x640 input",
        "results/rtdetr-l_coco.json, rtdetr-l_kitti.json", "consistent with, untested", "HYPOTHESIS")

    # ---- params ----
    D["params"] = {m: params[m]["params_M"] for m in MODELS}
    for m in MODELS:
        add(f"params_{m}", f"{LABEL[m]} parameters (M)", "results/params/model_params.json", f"{params[m]['params_M']}", "MEASURED")

    # ---- misses (score >= 0.4, IoU >= 0.5) ----
    D["miss"] = {m: {k: float(v) for k, v in miss[m].items() if k != "model"} for m in MODELS}
    for m in MODELS:
        add(f"miss_{m}", f"{LABEL[m]} KITTI miss rate small/medium/large then person/car/truck (score>=0.4, IoU>=0.5)",
            "figures/failures/miss_rates.csv",
            " / ".join(f"{float(miss[m][k]):.3f}" for k in SIZES) + " then " +
            " / ".join(f"{float(miss[m][k]):.3f}" for k in CLASSES), "MEASURED")

    # ---- hypotheses ----
    add("hyp_composition", "Car AP looks flat overall partly because the small-car share differs strongly between datasets (see car_small_share)",
        "results/class_size_ap.csv", "not isolated by an experiment", "HYPOTHESIS")
    add("hyp_resolution", "Faster R-CNN's smaller KITTI drop is partly due to its different effective input on KITTI (see input_shapes); the 1280 run tests this only for YOLO11s",
        "results/yolo11s_kitti_imgsz1280.json", "partially tested, not isolated", "HYPOTHESIS")
    add("hyp_unlabeled", "Many false positives in the selected failure figures are real but unlabeled objects (Ultralytics KITTI has no DontCare regions)",
        "figures/report/SELECTION.md", "visual judgement only", "HYPOTHESIS")
    add("bucket_note", "Size buckets differ across datasets: COCO area from segmentation masks, converted KITTI from box area",
        "annotations/kitti_val_coco.json, COCO instances_val2017.json", "by construction", "MEASURED")
    add("ar1_note", "AR@1 differs between the 80-class and 3-class tables because maxDets is capped per image per class; compare AR@100",
        "results/*_coco.json", "by construction", "MEASURED")
    return claims, D


def main():
    claims, D = compute()
    (RES / "claims.json").write_text(json.dumps(D, indent=2))
    lines = ["# Claims ledger", "",
             "Generated by `scripts/claims_check.py` from files in this repo. The report may only use claims listed here.",
             "Statuses: MEASURED = read directly from a result file; DERIVED = computed from result files; "
             "HYPOTHESIS = plausible explanation, not isolated by an experiment; UNVERIFIED = observed but not explained.", "",
             "| Claim | Source file | Value | Status |", "|---|---|---|---|"]
    for c in claims:
        lines.append(f"| {c['claim']} | {c['source']} | {c['value']} | {c['status']} |")
    (ROOT / "docs" / "claims_ledger.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    cnt = Counter(c["status"] for c in claims)
    print(f"{len(claims)} claims: {dict(cnt)}")
    print("Wrote results/claims.json and docs/claims_ledger.md")


if __name__ == "__main__":
    main()
