"""
Build the Track A report. Every number is computed from files in this repo
(results/*.json, results/class_size_ap.csv, figures/failures/miss_rates.csv,
docs/*) via scripts/claims_check.py; nothing is typed by hand.

Output: report/report.html, report/CMPE249_HW_2D_Detection_TrackA.pdf,
        report/preview/page_N.png
"""
import html, re, subprocess, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import claims_check as cc  # noqa: E402

ROOT = cc.ROOT
OUT = ROOT / "report"
OUT.mkdir(exist_ok=True)
REPO = "https://github.com/AdamSabry1233/2D-Object-Detection"
EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
M = cc.MODELS
L = cc.LABEL
M3 = cc.M3

R, R1280, PARAMS, PUB, CS, MISS, KGT = cc.load()
CLAIMS, D = cc.compute()


# ---------- helpers ----------
def f3(x): return f"{x:.3f}"
def p1(x): return f"{x:+.1f}%"
def st(m, ds, k="AP"): return R[(m, ds)]["metrics"][M3]["stats"][k]
def pcl(m, ds, c): return R[(m, ds)]["metrics"][M3]["per_class_AP"][c]
def esc(s): return html.escape(str(s))


def table(head, rows, cls="", note=None):
    h = "".join(f"<th>{c}</th>" for c in head)
    b = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<table class="{cls}"><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


def fig(src, caption, num, cls=""):
    return (f'<figure class="{cls}"><img src="{src}"><figcaption><b>Figure {num}.</b> {caption}</figcaption></figure>')


def tcap(num, text): return f'<div class="tcap"><b>Table {num}.</b> {text}</div>'


# ---------- hardware / software, read from files ----------
hw = (ROOT / "docs" / "hardware.txt").read_text(errors="ignore")
driver = re.search(r"KMD Version:\s*([\d.]+)", hw).group(1)
cap_w = re.search(r"/\s*(\d+)W", hw).group(1)
vram_gb = int(re.search(r"/\s*(\d+)MiB", hw).group(1)) / 1024
ram_gb = re.search(r"RAM_GB=([\d.]+)", hw).group(1)
r0 = R[("yolo11s", "coco")]
ver = r0["versions"]
gpu = r0["hardware"]

# ---------- references parsed from docs/references.md ----------
refs = {}
cur = None
for line in (ROOT / "docs" / "references.md").read_text(encoding="utf-8").splitlines():
    mm = re.match(r"^(\d+)\. (.*)", line)
    if mm:
        cur = int(mm.group(1))
        refs[cur] = {"text": mm.group(2), "url": None}
        continue
    um = re.match(r"^\s+URL:\s*(\S+)", line)
    if um and cur and refs[cur]["url"] is None:
        refs[cur]["url"] = um.group(1)
ref_html = []
for n in sorted(refs):
    t = refs[n]["text"]
    parts = re.split(r"\s\(fetched.*\)\.", t)
    t = parts[0] + (parts[1] if len(parts) > 1 else "")
    t = re.sub(r"^(.*?)(\s*https?://\S+)(.*)$", r"\1\3", t) if refs[n]["url"] is None else t
    urls = re.findall(r"https?://\S+", refs[n]["text"])
    url = refs[n]["url"] or (urls[0].rstrip(".,;)") if urls else "")
    ref_html.append(f'<li value="{n}">{esc(t.replace(url, "").replace("`", "").strip())} <span class="url">{esc(url)}</span></li>')

# ---------- tables ----------
# Table 1: models
t1 = table(
    ["Model", "Family / design", "Checkpoint", "Params (M)", "NMS", "Input handling"],
    [["YOLO11s", "single-stage CNN, anchor-free", "yolo11s.pt (Ultralytics)", PARAMS["yolo11s"]["params_M"],
      f"yes (IoU {R[('yolo11s','coco')]['config']['nms_iou']})", "imgsz 640, letterboxed"],
     ["RT-DETR-L", "transformer: hybrid encoder + query decoder", "rtdetr-l.pt (Ultralytics)", PARAMS["rtdetr-l"]["params_M"],
      "none", "imgsz 640, letterboxed"],
     ["Faster R-CNN R50-FPN", "two-stage, anchor-based (RPN + RoI head)", "torchvision COCO_V1", PARAMS["frcnn"]["params_M"],
      f"yes (IoU {R[('frcnn','coco')]['config']['nms_iou']})", "internal resize, min side 800 / max side 1333"]],
    "t")

# Table 2: datasets
nk = D["kitti_counts"]
t2 = table(
    ["Dataset", "Split", "Images", "Objects scored (person / car / truck)", "Notes"],
    [["COCO", "val2017", D["n_images"]["coco"], f"{D['size_dist']['coco']['n']} total (non-crowd)",
      "80 classes; 3 shared classes used for comparison"],
     ["KITTI", "val (Ultralytics repackaging)", D["n_images"]["kitti"],
      f"{nk['person']} / {nk['car']} / {nk['truck']} (+{nk['ignore']} ignore regions)",
      "converted to COCO format, COCO category ids"]], "t")

# Table 3: label mapping
t3 = table(
    ["KITTI class", "Mapped to", "Reason"],
    [["car", "car (3)", "direct match"],
     ["van", "car (3)", "COCO has no van class"],
     ["truck", "truck (8)", "direct match"],
     ["pedestrian", "person (1)", "direct match"],
     ["person_sitting", "person (1), ignore region (iscrowd=1)", "not scored: neither a hit nor a false positive"],
     ["cyclist", "person (1), ignore region (iscrowd=1)",
      "box covers rider plus bike, but COCO models predict person and bicycle separately; scoring it would create artificial false positives"],
     ["tram, misc", "dropped", "no matching class among the three scored"]], "t")

# Table 4: sanity check
mx = max(abs(D["ap80"][m]["diff"]) for m in M)
t4 = table(
    ["Model", "Measured AP (80 classes)", "Published AP", "Difference", "Published source"],
    [[L[m], f3(D["ap80"][m]["measured"]), f3(D["ap80"][m]["published"]), f"{D['ap80'][m]['diff']:+.3f}", src]
     for m, src in zip(M, ["[10]", "[11]", "[13] (weights metadata)"])], "t")

# Table 5: headline
rows5 = []
for m in M:
    for ds in ("coco", "kitti"):
        s = R[(m, ds)]["metrics"][M3]["stats"]
        rows5.append([L[m], ds.upper(), f3(s["AP"]), f3(s["AP50"]), f3(s["AP75"]), f3(s["AP_small"]), f3(s["AP_medium"]),
                      f3(s["AP_large"]), f3(s["AR100"]), f3(pcl(m, ds, "person")), f3(pcl(m, ds, "car")), f3(pcl(m, ds, "truck"))])
s = R1280["metrics"][M3]["stats"]
rows5.append(["YOLO11s, imgsz 1280 (resolution exp.)", "KITTI", f3(s["AP"]), f3(s["AP50"]), f3(s["AP75"]), f3(s["AP_small"]),
              f3(s["AP_medium"]), f3(s["AP_large"]), f3(s["AR100"]), f3(R1280["metrics"][M3]["per_class_AP"]["person"]),
              f3(R1280["metrics"][M3]["per_class_AP"]["car"]), f3(R1280["metrics"][M3]["per_class_AP"]["truck"])])
t5 = table(["Model", "Data", "AP", "AP50", "AP75", "APs", "APm", "APl", "AR100", "AP person", "AP car", "AP truck"], rows5, "t small")

# Table 6: latency
rows6 = []
for m in M:
    for ds in ("coco", "kitti"):
        r = R[(m, ds)]
        rows6.append([L[m], ds.upper(), r["config"]["input_size"].split(" ")[0] if m != "frcnn" else "internal 800/1333",
                      r["latency_ms"]["mean"], r["latency_ms"]["median"], r["latency_ms"]["p95"], r["fps"], r["peak_gpu_mem_gb"]])
t6 = table(["Model", "Data", "Input", "Mean ms", "Median ms", "p95 ms", "FPS", "Peak GPU mem (GB)"], rows6, "t")

# Table 7: class x size
rows7 = []
for c in cc.CLASSES:
    for sz in cc.SIZES:
        cell = {m: next(x for x in D["cells"] if x["model"] == m and x["cls"] == c and x["size"] == sz) for m in M}
        x0 = cell["yolo11s"]
        flag = "" if x0["reliable"] else "&dagger;"
        rows7.append([c, sz + flag, x0["n_coco"], x0["n_kitti"]] +
                     [f"{f3(cell[m]['ap_coco'])} &rarr; {f3(cell[m]['ap_kitti'])}" for m in M])
t7 = table(["Class", "Size", "n COCO", "n KITTI"] + [L[m] for m in M], rows7, "t small")

# Table 8: miss rates
t8 = table(["Model", "small", "medium", "large", "person", "car", "truck"],
           [[L[m]] + [f3(D["miss"][m][k]) for k in cc.SIZES + cc.CLASSES] for m in M], "t")

# ---------- derived values used in text ----------
do = D["drop_overall"]
dc = D["drop_class"]
ds_ = D["drop_size"]
rk = D["rank"]
res = D["res1280"]
rs = D["resolution"]
sp = D["speed"]
cd = D["car_cells_dropping"]
fk = D["frcnn_kitti_truck"]
sd = D["size_dist"]
cs_share = D["car_small_share"]
lat = D["lat"]
pdrop = D["person_drop_size"]
ratio = D["ap75_over_ap50"]
frcnn_large_car = next(x for x in D["cells"] if x["model"] == "frcnn" and x["cls"] == "car" and x["size"] == "large")
rank_txt = {ds: " &gt; ".join(L[m] for m in rk[ds]) for ds in ("coco", "kitti")}
pd_min = min(pdrop.values()); pd_max = max(pdrop.values())
sp_person_range = (min(D["kitti_small_person_ap"].values()), max(D["kitti_small_person_ap"].values()))

# ---------- guard: narrative claims must match the data, or the build fails ----------
assert rk["coco"] == ["rtdetr-l", "yolo11s", "frcnn"] and rk["kitti"] == ["rtdetr-l", "frcnn", "yolo11s"], "ranking text"
assert all(ratio[m][1] < ratio[m][0] for m in M), "AP75/AP50 should fall on KITTI for all models"
assert all(v < 0 for v in pdrop.values()), "person AP should drop at every size"
assert all(dc[f"{m}_person"] < -40 and dc[f"{m}_truck"] < -40 for m in M), "person/truck drop strongly"
assert all(abs(dc[f"{m}_car"]) < 10 for m in M), "car flat overall"
assert cd[0] == cd[1], "car drops in all reliable cells"
assert all(sp[ds]["rtdetr_faster_and_at_least_as_accurate_as_frcnn"] for ds in ("coco", "kitti")), "RT-DETR vs FRCNN"
assert all(max(D["miss"][m], key=D["miss"][m].get) == "truck" for m in M), "trucks missed most often"
assert all(lat[f"frcnn_{ds}"]["mem"] > lat[f"rtdetr-l_{ds}"]["mem"] > lat[f"yolo11s_{ds}"]["mem"] for ds in ("coco", "kitti")), "memory order"
assert all(lat[f"yolo11s_{ds}"]["fps"] > lat[f"rtdetr-l_{ds}"]["fps"] > lat[f"frcnn_{ds}"]["fps"] for ds in ("coco", "kitti")), "fps order"
assert abs(lat["yolo11s_change_pct"]) < 20 and abs(lat["rtdetr-l_change_pct"]) < 20 and abs(lat["frcnn_change_pct"]) > 20, "latency noise text"
assert res["gap_to_frcnn"] < 0.01 < res["gap_to_rtdetr"], "1280 text"
assert D["car_small_share"]["coco"] > 50 > D["car_small_share"]["kitti"], "composition text"
assert sd["kitti"]["small"] < sd["coco"]["small"], "KITTI does not have more small objects"
assert max(abs(D["ap80"][m]["diff"]) for m in M) < 0.02, "sanity text"
assert D["miss"]["yolo11s"]["small"] > D["miss"]["frcnn"]["small"], "miss-rate text"

# ---------- failure figures ----------
FAIL = [
    ("missed_by_all_1_005226", "005226",
     "<i>Missed by all.</i> Large vehicle at the left edge is labelled car but predicted as truck by all three models (red boxes; their labels are partly cropped in this compact view, see the full-size original in <code>figures/report/</code>); a crowd of small, overlapping pedestrians is mostly missed, while the two close pedestrians on the right are found by all."),
    ("missed_by_all_4_005245", "005245",
     "<i>Missed by all.</i> Parked cars with heavy overlap, one car truncated by the image border, and distant cars are missed by all three models; Faster R-CNN adds many small red boxes on the far right."),
    ("disagreement_1_003554", "003554",
     "<i>Disagreement.</i> A row of distant cars: YOLO11s finds 1 and misses 13, RT-DETR-L finds 10, Faster R-CNN finds 8 but adds 14 false positives."),
    ("disagreement_2_003219", "003219",
     "<i>Disagreement.</i> Dense parked cars: Faster R-CNN finds the most (16 correct, 4 missed) and has the most false positives (19); most of its red boxes sit on small distant cars."),
    ("false_positives_2_000018", "000018",
     "<i>False positives.</i> Almost empty road with 2 labelled cars; the many red car boxes sit on what look like distant parked cars that KITTI did not label."),
    ("false_positives_3_006255", "006255",
     "<i>False positives.</i> Same kind of scene; besides red car boxes on distant parked cars, YOLO11s and Faster R-CNN place red person boxes on thin upright roadside objects, which may be a real model error (I cannot tell what they are)."),
]
fail_html = []
for i, (name, iid, text) in enumerate(FAIL):
    num = 4 + i
    idx = next(r for r in (ROOT / "figures" / "failures" / "index.csv").read_text().splitlines() if iid in r).split(",")
    cap = (f"KITTI {iid}. {text} Matching at score &ge; 0.4 and IoU &ge; 0.5. Blue = correct, red = false positive "
           f"(class and score), orange = missed ground truth, dashed gray = ignore region. Panels are cropped vertically to the band "
           f"containing the boxes, and class/score labels are drawn only on large boxes; image chosen as an extreme case of its category, not a typical one, and some red boxes may be real but unlabelled objects.")
    fail_html.append(fig(f"../figures/report/compact/{name}.png", cap, num, "fail"))

# ---------- body ----------
body = f"""
<h1>2D Object Detection with COCO-Pretrained Detectors: Generalization from COCO to KITTI (Track A)</h1>
<div class="by">Adam Sabry &middot; CMPE-249 Sec 33, Fall 2026<br>Code, results and command log: <span class="url">{REPO}</span></div>

<h2>1. Objective and approach</h2>
<p>I ask how well three COCO-pretrained detectors from different architecture families transfer to KITTI driving scenes <i>without retraining</i>, and what each costs in speed and memory. I evaluate YOLO11s, RT-DETR-L and Faster R-CNN R50-FPN on COCO val2017 and the KITTI val split, restricted to the three classes both datasets share (person, car, truck), and score every run with pycocotools. The two analyses are cross-dataset generalization (Section 5) and the speed-accuracy trade-off with a short architecture discussion (Section 6); Section 7 is a failure analysis. Every number below is computed from files in the repository; <code>docs/claims_ledger.md</code> lists each claim with its source file and a status (measured, derived, hypothesis or unverified), and I label statements as measured or hypothesis in the text. AP values are fractions with three decimals.</p>

<h2>2. Models and datasets</h2>
<p>Table 1 lists the models, Table 2 the datasets (COCO [1]; KITTI [2], [3], [15]) and Table 3 the KITTI-to-COCO label mapping used to compare them.</p>
{tcap(1, "The three evaluated detectors. Parameter counts are from <code>results/params/model_params.json</code>. YOLO11 has no paper; its documentation is the citation [10]. Architecture references: [4]&ndash;[9].")}
{t1}
{tcap(2, "Datasets. COCO is scored on the full val2017 split; KITTI on the 1,496-image val split of the Ultralytics repackaging [15], converted to COCO format by <code>scripts/convert_kitti_to_coco.py</code>.")}
{t2}
{tcap(3, "KITTI-to-COCO label mapping. The cross-dataset comparison uses person (COCO id 1), car (3) and truck (8).")}
{t3}

<h2>3. Experimental setup, hardware and reproducibility</h2>
<p><b>Hardware and software.</b> {esc(gpu)} ({vram_gb:.0f} GB VRAM, {cap_w} W power cap, driver {driver}), {ram_gb} GB RAM, Windows 11 (Git Bash). Python {r0['python']}, torch {ver['torch']}, torchvision {ver['torchvision']}, ultralytics {ver['ultralytics']}, pycocotools {ver['pycocotools']}, numpy {ver['numpy']}. These come from the <code>hardware</code> and <code>versions</code> fields stored in every result file.</p>
<p><b>Protocol.</b> YOLO11s and RT-DETR-L run through the Ultralytics package [12], Faster R-CNN through torchvision [13], and all predictions are scored with pycocotools [14]. Confidence threshold {r0['config']['conf_thresh']} and at most {r0['config']['max_det']} detections per image for all models, so that AP is computed over the full precision-recall curve. NMS IoU is {R[('yolo11s','coco')]['config']['nms_iou']} for YOLO11s, {R[('frcnn','coco')]['config']['nms_iou']} for Faster R-CNN, and none for RT-DETR-L. YOLO11s and RT-DETR-L run at imgsz 640 (letterboxed); Faster R-CNN uses torchvision's internal resize (min side 800, max side 1333), so the models see KITTI at different scales (Section 5.5). Ultralytics class ids 0&ndash;79 are mapped to COCO ids with <code>COCO80_TO_91</code>. <b>Latency</b> is the mean time per image of preprocessing + model + NMS + copy to CPU, measured after 10 warmup images with CUDA synchronization; image decoding is not timed. <b>Peak memory</b> is <code>torch.cuda.max_memory_allocated</code> (tensor allocations only, excluding the CUDA context). Because the GPU is a power-limited laptop part, I treat latency differences under about 20% as noise.</p>
<p><b>Reproduction and evidence.</b> Each <code>results/&lt;model&gt;_&lt;dataset&gt;.json</code> stores the metrics, latency, memory, configuration, versions, hardware and the exact command that produced it; <code>COMMANDS.md</code> lists all commands in order, and the code is in <code>scripts/</code> of the repository above. Raw detections (large) and datasets are kept outside the repository. A 50-image smoke test is excluded from all results. No training was done, so the KITTI val split carries no leakage concern.</p>

<h2>4. Quantitative results</h2>
<p><b>Sanity check.</b> Table 4 compares my 80-class COCO AP with the published values. All three agree within {mx:.3f} AP, which I take as evidence that the label mapping and scoring pipeline are correct. YOLO11s has the largest gap ({D['ap80']['yolo11s']['diff']:+.3f}); a possible cause is that my per-image <code>predict</code> settings differ from the vendor's validation settings (hypothesis, not tested). The published YOLO11s page does not name the split [10].</p>
{tcap(4, "80-class COCO AP@[.50:.95] on val2017, measured vs. published.")}
{t4}
<p><b>Main results.</b> Table 5 gives the 3-class results and Table 6 the cost. Figure 1 shows the headline result: every detector loses a large share of its AP on KITTI.</p>
{tcap(5, "3-class (person, car, truck) results. AP = AP@[.50:.95]; s/m/l = small/medium/large objects; AR100 = recall with up to 100 detections. The last row is the separate resolution experiment (Section 5.5); the model is the same YOLO11s checkpoint. Truck on KITTI has only {nk['truck']} instances, so its AP is noisy.".replace("{nk['truck']}", str(nk['truck'])))}
{t5}
{fig("../figures/fig1_coco_vs_kitti_ap.png", f"3-class AP@[.50:.95] on COCO val2017 and KITTI val for the three COCO-pretrained detectors (no retraining). Percentages are the relative change COCO &rarr; KITTI. Conf 0.001, max 100 detections, NMS IoU as in Table 1. Caveat: models see different input scales and no confidence intervals were computed, so AP gaps of about {abs(D['gap_rtdetr_frcnn_kitti']):.2f} between models are not established.", 1, "narrow")}
{tcap(6, "Latency, throughput and peak GPU memory (mean over images after 10 warmup images; 45 W laptop GPU, differences under about 20% treated as noise).".replace("45 W", f"{cap_w} W"))}
{t6}

<h2>5. Analysis 1: cross-dataset generalization</h2>
<p><b>5.1 Overall.</b> <i>Expected:</i> a drop on a different domain, with the COCO ranking roughly preserved. <i>Observed (derived):</i> 3-class AP falls by {abs(do['yolo11s']):.1f}% for YOLO11s, {abs(do['rtdetr-l']):.1f}% for RT-DETR-L and {abs(do['frcnn']):.1f}% for Faster R-CNN (Figure 1), and the ranking changes: {rank_txt['coco']} on COCO, but {rank_txt['kitti']} on KITTI. RT-DETR-L and Faster R-CNN differ by {D['gap_rtdetr_frcnn_kitti']:.3f} AP on KITTI ({f3(st('rtdetr-l','kitti'))} vs {f3(st('frcnn','kitti'))}); without confidence intervals I treat this as too close to call. The AP75/AP50 ratio also falls for all models on KITTI ({ratio['yolo11s'][0]:.2f} &rarr; {ratio['yolo11s'][1]:.2f}, {ratio['rtdetr-l'][0]:.2f} &rarr; {ratio['rtdetr-l'][1]:.2f}, {ratio['frcnn'][0]:.2f} &rarr; {ratio['frcnn'][1]:.2f} for YOLO11s, RT-DETR-L, Faster R-CNN), so objects that pass IoU 0.5 are less often tight enough to pass 0.75 (measured); why is untested.</p>
<p><b>5.2 By class.</b> Figure 2 shows that the drop is concentrated in two classes. Person AP changes by {p1(dc['yolo11s_person'])}, {p1(dc['rtdetr-l_person'])} and {p1(dc['frcnn_person'])} and truck AP by {p1(dc['yolo11s_truck'])}, {p1(dc['rtdetr-l_truck'])} and {p1(dc['frcnn_truck'])} (YOLO11s, RT-DETR-L, Faster R-CNN), while car AP changes by only {p1(dc['yolo11s_car'])}, {p1(dc['rtdetr-l_car'])} and {p1(dc['frcnn_car'])}. The truck numbers rest on {nk['truck']} objects and are noisy; I did not test whether label-definition differences (for example vans merged into car) contribute.</p>
{fig("../figures/fig2_per_class_ap.png", f"Per-class AP@[.50:.95] (3-class) on COCO and KITTI. Person and truck degrade; car is flat overall (but see Table 7: at matched sizes car AP is lower on KITTI). Truck n={nk['truck']} on KITTI, so its AP is noisy; KITTI person n={nk['person']}. Same detection thresholds as Figure 1.", 2)}
<p><b>5.3 By size and class &times; size.</b> Table 7 separates class from size. At matched sizes car AP is <i>lower</i> on KITTI in {cd[0]} of {cd[1]} reliable cells, even though pooled car AP looks flat; the closest to flat is Faster R-CNN on large cars ({f3(frcnn_large_car['ap_coco'])} vs {f3(frcnn_large_car['ap_kitti'])}, {p1(frcnn_large_car['drop_pct'])}). Person AP drops at every size, by {abs(pd_max):.0f}% to {abs(pd_min):.0f}% across models and sizes, and small KITTI persons score only {sp_person_range[0]:.3f}&ndash;{sp_person_range[1]:.3f} AP for all three models, so pedestrians fail at every size (a class-level problem, not only a size problem; measured).</p>
{tcap(7, "AP by class and object size, COCO &rarr; KITTI, with object counts. Size buckets are COCO's (small &lt; 32&sup2; px, medium &lt; 96&sup2; px) but COCO area comes from segmentation masks and the converted KITTI file uses box area. &dagger; fewer than 50 objects in one dataset: unreliable.")}
{t7}
<p><b>5.4 Composition effect.</b> The flat pooled car AP is consistent with a composition effect (hypothesis, not isolated by an experiment): {cs_share['coco']:.0f}% of COCO cars are small versus {cs_share['kitti']:.0f}% of KITTI cars, and small objects have much lower AP. KITTI does <i>not</i> have more small objects than COCO in this 3-class view: person/car/truck are {sd['coco']['small']:.0f}/{sd['coco']['medium']:.0f}/{sd['coco']['large']:.0f}% small/medium/large on COCO and {sd['kitti']['small']:.0f}/{sd['kitti']['medium']:.0f}/{sd['kitti']['large']:.0f}% on KITTI (measured). Pooled size AP therefore needs care: Faster R-CNN's pooled small-object AP is almost unchanged ({f3(st('frcnn','coco','AP_small'))} &rarr; {f3(st('frcnn','kitti','AP_small'))}, {p1(ds_['frcnn_small'])}), but this is driven by {fk['n_small']} small KITTI trucks (small-truck AP {f3(fk['small'])} versus medium {f3(fk['medium'])}), which is noise. I therefore do <b>not</b> claim that Faster R-CNN is robust on small objects.</p>
<p><b>5.5 Resolution confound and the 1280 experiment.</b> For the most common KITTI image size ({rs['w']}&times;{rs['h']}), the 640 letterbox scales the image by {rs['scale_ultralytics']:.2f}, while Faster R-CNN's resize scales it by {rs['scale_frcnn']:.2f} (to about {rs['frcnn_hw'][1]}&times;{rs['frcnn_hw'][0]}), about {rs['ratio']:.1f}&times; the linear scale (derived). This confounds any architecture-versus-architecture claim on KITTI. To test it for one model, I re-ran YOLO11s on KITTI at imgsz 1280: 3-class AP rose from {f3(res['ap_640'])} to {f3(res['ap_1280'])} ({res['delta']:+.3f}, {p1(res['delta_pct'])}) and small-object AP from {f3(res['ap_small_640'])} to {f3(res['ap_small'])}. YOLO11s then trails RT-DETR-L by {res['gap_to_rtdetr']:.3f} AP and Faster R-CNN by only {res['gap_to_frcnn']:.3f}, within what I consider noise. So resolution explains part of YOLO11s' KITTI deficit relative to Faster R-CNN, but it does not close the gap to RT-DETR-L. I did not vary resolution for the other two models, and the 1280 latency ({res['latency_ms']} ms) is not comparable to the 640 runs; I do not interpret it.</p>

<h2>6. Analysis 2: speed-accuracy trade-off and architecture</h2>
<p>Figure 3 plots AP against latency. <i>Measured:</i> YOLO11s is the fastest and smallest ({lat['yolo11s_coco']['fps']} FPS on COCO, {lat['yolo11s_kitti']['fps']} on KITTI, {lat['yolo11s_coco']['mem']} GB peak). On COCO, RT-DETR-L has {sp['coco']['rtdetr_over_yolo_latency']:.1f}&times; YOLO11s' latency for {sp['coco']['ap_gain_rtdetr_over_yolo']:+.3f} AP ({p1(sp['coco']['ap_gain_rtdetr_over_yolo_pct'])}); on KITTI the AP gain is {sp['kitti']['ap_gain_rtdetr_over_yolo']:+.3f} ({p1(sp['kitti']['ap_gain_rtdetr_over_yolo_pct'])}) at {sp['kitti']['rtdetr_over_yolo_latency']:.1f}&times; the latency. Faster R-CNN is the slowest ({lat['frcnn_coco']['fps']} FPS on COCO, {lat['frcnn_kitti']['fps']} on KITTI) and uses the most memory. RT-DETR-L is faster ({sp['coco']['frcnn_over_rtdetr_latency']:.1f}&times; on COCO, {sp['kitti']['frcnn_over_rtdetr_latency']:.1f}&times; on KITTI) and at least as accurate as Faster R-CNN on both datasets ({sp['coco']['ap_rtdetr_minus_frcnn']:+.3f} and {sp['kitti']['ap_rtdetr_minus_frcnn']:+.3f} AP), although the KITTI AP gap is small (Section 5.1), so RT-DETR-L is the most balanced choice here and YOLO11s the choice if speed or memory dominate.</p>
{fig("../figures/fig3_speed_accuracy.png", f"3-class AP@[.50:.95] versus mean latency per image on COCO (left) and KITTI (right). Latency = preprocessing + model + NMS + copy to CPU, mean after 10 warmup images, on a {cap_w} W laptop GPU; differences under about 20% are treated as noise. The y-axis is truncated at 0.22 to separate the points. No error bars (single run per configuration).", 3)}
<p><b>Latency across datasets.</b> From COCO to KITTI the mean latency changes by {lat['yolo11s_change_pct']:+.0f}% for YOLO11s, {lat['rtdetr-l_change_pct']:+.0f}% for RT-DETR-L and {lat['frcnn_change_pct']:+.0f}% for Faster R-CNN (Table 6). Only Faster R-CNN's change is larger than my noise threshold; a plausible cause is fewer pixels after torchvision's resize on KITTI's wide, short images (hypothesis, untested). YOLO11s' small increase is unexplained, and RT-DETR-L's near-identical latency and memory on both datasets are consistent with its fixed 640&times;640 input (hypothesis).</p>
<p><b>Architecture discussion (hedged).</b> YOLO11s is a single-stage, anchor-free CNN with NMS on the output; RT-DETR-L is query-based with a transformer decoder and needs no NMS [7][8]; Faster R-CNN is two-stage and anchor-based with an RPN and NMS [4][5][6]. The measurements are consistent with the usual expectation that the one-stage model is cheapest and the two-stage model costs the most per image. They do not isolate the cause of RT-DETR-L's better accuracy on both datasets: its checkpoint, parameter count ({PARAMS['rtdetr-l']['params_M']} M vs {PARAMS['yolo11s']['params_M']} M for YOLO11s) and training recipe all differ from YOLO11s, so I cannot attribute the gap to the NMS-free design (hypothesis). Likewise the comparison with Faster R-CNN is confounded by input scale (Section 5.5).</p>

<h2>7. Failure analysis</h2>
<p>I matched detections to ground truth at score &ge; 0.4 and IoU &ge; 0.5 (<code>scripts/failure_analysis.py</code>) and ranked KITTI images by misses common to all models (&ldquo;missed by all&rdquo;), by the spread in misses between models (&ldquo;disagreement&rdquo;), and by total false positives. I picked two of the four top images per category that make different points; the reasons are in <code>figures/report/SELECTION.md</code>. These are extreme cases by construction. Table 8 gives the corresponding overall miss rates: YOLO11s misses {D['miss']['yolo11s']['small']:.0%} of small objects versus {D['miss']['frcnn']['small']:.0%} for Faster R-CNN, and trucks are missed most often by every model.</p>
{tcap(8, "KITTI miss rate (fraction of ground-truth objects with no matching detection) at score &ge; 0.4, IoU &ge; 0.5, by object size and class.")}
{t8}
<p>Figure 4 shows a vehicle labelled <i>car</i> that all models call <i>truck</i> and a crowd of small pedestrians that is mostly missed, consistent with a class-definition mismatch and small, overlapping people (the vehicle's original KITTI class is not visible in the image). Figure 5 shows overlap and truncation among parked cars. Figures 6 and 7 show where the models disagree: YOLO11s misses most distant cars that RT-DETR-L finds (Figure 6), and Faster R-CNN finds the most cars but adds many extra boxes (Figure 7), consistent with the higher input scale and miss-rate pattern in Table 8 (hypothesis). Figures 8 and 9 show false positives: many red car boxes sit on distant parked cars that appear real but unlabelled (the Ultralytics KITTI export has no DontCare regions), so KITTI AP may understate precision (hypothesis, from visual inspection only). The red person boxes in Figure 9 may be genuine errors; I cannot tell what the objects are at this size.</p>
{"".join(fail_html)}

<h2>8. Limitations and threats to validity</h2>
<ul>
<li><b>Labels.</b> The Ultralytics KITTI export drops DontCare regions and has no truncation or occlusion flags, so some false positives may be real unlabelled objects. Van is merged into car; cyclist and sitting person are ignore regions; tram and misc are dropped. Only {nk['truck']} trucks exist in KITTI val.</li>
<li><b>Resolution.</b> Models are evaluated at their default scales; Faster R-CNN sees KITTI at about {rs['ratio']:.1f}&times; the linear scale of the other two. Only YOLO11s was re-run at a higher resolution.</li>
<li><b>Statistics.</b> One run per configuration, no confidence intervals or bootstrap, so small AP differences between models are not established.</li>
<li><b>Size buckets.</b> COCO area is mask area, KITTI area is box area, so the buckets are not identical across datasets.</li>
<li><b>Latency.</b> Measured on a power-limited laptop GPU, excluding image decoding; differences under about 20% are noise. The 1280 latency is not comparable to the 640 runs.</li>
<li><b>Failure figures.</b> Selected extreme cases, judged visually; causes are hypotheses.</li>
<li><b>AR@1</b> differs between the 80-class and 3-class tables because maxDets is capped per image per class; AR@100 is the meaningful comparison.</li>
</ul>

<h2>9. Conclusions</h2>
<ul>
<li>All three COCO-pretrained detectors lose a large share of their 3-class AP on KITTI ({abs(do['yolo11s']):.0f}%, {abs(do['rtdetr-l']):.0f}%, {abs(do['frcnn']):.0f}% for YOLO11s, RT-DETR-L, Faster R-CNN), and the ranking changes (Section 5.1).</li>
<li>The loss is concentrated in person and truck; the flat pooled car AP hides lower AP at matched sizes and is consistent with a composition effect (Sections 5.2&ndash;5.4).</li>
<li>Faster R-CNN's smaller relative drop and its pooled small-object AP should not be read as robustness: it is partly an input-scale confound and small-truck noise. Raising YOLO11s to 1280 recovers part of its deficit ({p1(res['delta_pct'])}).</li>
<li>RT-DETR-L gives the best accuracy-speed balance here; YOLO11s is the cheapest. The KITTI gap between RT-DETR-L and Faster R-CNN is within what I cannot resolve without confidence intervals.</li>
<li>Many KITTI &ldquo;false positives&rdquo; appear to be real unlabelled objects, so KITTI AP is a conservative estimate (hypothesis). Fine-tuning on KITTI and per-class statistics with confidence intervals are the obvious next steps.</li>
</ul>

<h2>10. References</h2>
<ol class="refs">{"".join(ref_html)}</ol>
<p class="ai">AI assistants (Claude) were used to help write code and draft this report; I ran all experiments and reviewed all results.</p>
"""

CSS = """
@page { size: Letter; margin: 0.55in 0.6in; }
* { box-sizing: border-box; }
body { font-family: "Segoe UI", Arial, sans-serif; font-size: 10.5pt; line-height: 1.3; color: #111; margin: 0; }
h1 { font-size: 16.5pt; line-height: 1.2; margin: 0 0 3pt 0; }
.by { font-size: 9.5pt; color: #333; margin-bottom: 6pt; }
h2 { font-size: 12.5pt; margin: 9pt 0 3pt 0; border-bottom: 1px solid #999; padding-bottom: 1pt; break-after: avoid; }
p { margin: 3pt 0 5pt 0; text-align: justify; }
code { font-family: Consolas, monospace; font-size: 9pt; }
table.t { border-collapse: collapse; width: 100%; font-size: 8.4pt; margin: 2pt 0 7pt 0; break-inside: avoid; }
table.t.small { font-size: 8pt; }
th, td { border: 1px solid #aaa; padding: 1.5pt 3.5pt; text-align: left; vertical-align: top; }
th { background: #e8e8e8; }
td:not(:first-child) { }
.tcap { font-size: 8.8pt; margin: 5pt 0 1pt 0; break-after: avoid; }
figure { margin: 4pt 0 6pt 0; break-inside: avoid; }
figure img { width: 100%; display: block; }
figure.narrow img { width: 82%; margin: 0 auto; }
figcaption { font-size: 8.6pt; margin-top: 2pt; text-align: justify; }
ul { margin: 2pt 0 4pt 0; padding-left: 15pt; } li { margin-bottom: 2pt; text-align: justify; }
ol.refs { font-size: 8.4pt; padding-left: 17pt; margin: 2pt 0; } ol.refs li { margin-bottom: 1.5pt; text-align: left; }
.url { font-family: Consolas, monospace; font-size: 7.8pt; overflow-wrap: anywhere; color: #1a3d7c; }
.ai { font-size: 9pt; margin-top: 5pt; }
"""

page = f'<!doctype html><html><head><meta charset="utf-8"><title>2D Object Detection, Track A</title><style>{CSS}</style></head><body>{body}</body></html>'
(OUT / "report.html").write_text(page, encoding="utf-8")
pdf = OUT / "CMPE249_HW_2D_Detection_TrackA.pdf"
if pdf.exists():
    pdf.unlink()
prof = tempfile.mkdtemp()
subprocess.run([EDGE, "--headless", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={prof}",
                f"--print-to-pdf={pdf}", (OUT / "report.html").resolve().as_uri()],
               check=True, capture_output=True, timeout=180)

import pymupdf  # noqa: E402
doc = pymupdf.open(pdf)
prev = OUT / "preview"
prev.mkdir(exist_ok=True)
for old in prev.glob("page_*.png"):
    old.unlink()
for i, pg in enumerate(doc, 1):
    pg.get_pixmap(dpi=80).save(prev / f"page_{i}.png")
print(f"Wrote {pdf.name}: {len(doc)} pages (limit 6-10), previews in report/preview/")
