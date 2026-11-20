"""Print markdown tables from whatever summary_*_K*.json files exist (works while jobs are still running).
    python make_table.py ~/eval_results_newSplit        (or the folder you scp'd back)
"""
import glob, json, sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else "eval_results_newSplit").expanduser()
rows = [json.load(open(f)) for f in glob.glob(str(d / "summary_*_K*.json"))]
if not rows:
    sys.exit(f"no summary_*_K*.json in {d}")

order = ["CFM-NewSplit", "RoPE-OG", "RoPE-TS", "RoPE-TS-Impose"]
for K in sorted({r["K"] for r in rows}):
    print(f"\n### K={K} (best of {K}), test split\n"
          "| model | epoch | minADE | minFDE | median minFDE | p90 | meanADE | meanFDE |\n|---|---|---|---|---|---|---|---|")
    for r in sorted((r for r in rows if r["K"] == K), key=lambda r: order.index(r["model"]) if r["model"] in order else 99):
        print(f"| {r['model']} | {r['epoch']} | {r['minADE']:.1f} | {r['minFDE']:.1f} | {r['minFDE_median']:.1f} | "
              f"{r['minFDE_p90']:.1f} | {r['meanADE']:.1f} | {r['meanFDE']:.1f} |")
print("\nbaselines:", rows[0]["baselines"])