"""Collect every results/*.json into one comparison table (printed + results/summary.csv)."""
import csv, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
rows = []
for f in sorted((ROOT / "results").glob("*.json")):
    if "limit" in f.stem:
        continue  # smoke tests
    d = json.loads(f.read_text())
    if "metrics" not in d:
        continue  # not an eval result file
    m = d["metrics"]["3class_person_car_truck"]
    s, pc = m["stats"], m["per_class_AP"]
    full = d["metrics"].get("coco_80class", {}).get("stats", {}).get("AP")
    rows.append({
        "model": d["model"], "dataset": d["dataset"],
        "input": d["config"]["input_size"].split(" ")[0],
        "AP80": full, "AP": s["AP"], "AP50": s["AP50"], "AP75": s["AP75"],
        "APs": s["AP_small"], "APm": s["AP_medium"], "APl": s["AP_large"],
        "AR100": s["AR100"],
        "AP_person": pc.get("person"), "AP_car": pc.get("car"), "AP_truck": pc.get("truck"),
        "lat_ms": d["latency_ms"]["mean"], "FPS": d["fps"], "mem_GB": d["peak_gpu_mem_gb"],
    })

cols = list(rows[0].keys())
print("  ".join(f"{c:>9}" for c in cols))
for r in rows:
    print("  ".join(f"{(r[c] if r[c] is not None else '-')!s:>9}" for c in cols))

with open(ROOT / "results" / "summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols)
    w.writeheader()
    w.writerows(rows)
print("\nWrote results/summary.csv")