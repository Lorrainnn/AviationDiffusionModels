"""
evalImposeSavio.py - evaluate the token-imposition model with k = 0,1,2,3 imposed future points.

Per window and per k, k random future positions (never the last one, so FDE stays honest) are
revealed at their TRUE values and held fixed during Euler sampling. Metrics:

  *_imp : error of the imposed model, ADE averaged over the FREE (non-revealed) points only
          (revealed points have zero error and would flatter ADE), FDE at the last point.
  *_ref : error of the same model with NOTHING imposed (k=0 samples), ADE averaged over the SAME
          free points. This is the paired baseline: same windows, same positions, same free set.
  imp - ref  =  how much revealing those k points helped (negative = helped).

k=0 has imp == ref (all 43 points free). K = samples per window (best-of-K / mean over K).

    python -u evalImposeSavio.py --nc_path $NC --Ks 5 --ks 0,1,2,3 --out $HOME/eval_results_newSplit
    python evalImposeSavio.py --table --out $HOME/eval_results_newSplit      # print finished tables

Saves summary_impose_K<K>.json and impose_K<K>.npz as soon as a K finishes.
"""
import argparse, importlib, json, sys, time, glob
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

MODULE  = "models.ditRoPETS_impose"
PATTERN = "checkpoints_cfm_ts_impose_newSplit*/best*.pt"
NAMES   = ["ade_imp", "ade_ref", "fde_imp", "fde_ref",
           "mean_ade_imp", "mean_ade_ref", "mean_fde_imp", "mean_fde_ref"]


def load_model(ckpt_path, device):
    mod = importlib.import_module(MODULE)
    model = mod.TrajectoryDiT(d_model=256, n_heads=8, n_layers=6).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["ema_state"])
    model.eval()
    return model, {"epoch": int(ckpt["epoch"]) + 1, "max_k": ckpt.get("max_k"),
                   "params": sum(p.numel() for p in model.parameters())}


@torch.no_grad()
def evaluate(model, loader, std, device, K, ks, steps, max_batches, seed, max_k):
    from models.cfm_impose import sample_impose_mask, euler_sample_impose
    torch.manual_seed(seed)
    acc = {k: {n: [] for n in NAMES} for k in ks}
    t0, n_b = time.time(), len(loader)
    dev = str(device)

    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs, fut, trel = b["obs"].to(device), b["fut"].to(device), b["t_rel"].to(device)
        B, L = fut.shape[0], fut.shape[1]
        gt = (fut[..., :2] * std)[None]
        err_of = lambda p: torch.norm(p[..., :2] * std - gt, dim=-1)          # (K,B,L) metres

        none = torch.zeros(B, L, dtype=torch.bool, device=device)
        e0 = err_of(euler_sample_impose(model, obs, trel, fut, none, n_samples=K, n_steps=steps, device=dev))

        for k in ks:
            if k == 0:
                m, ek = none, e0                                              # same samples, nothing imposed
            else:
                m = sample_impose_mask(B, L, max_k=max_k, device=device, allow_last=False, k=k)
                ek = err_of(euler_sample_impose(model, obs, trel, fut, m, n_samples=K, n_steps=steps, device=dev))
                if i == 0:
                    assert float(ek[:, m].max()) < 1e-2, "imposed points moved during sampling"
                    assert not m[:, -1].any(), "last point must never be imposed"
            free  = (~m).float()                                              # (B,L)
            nfree = free.sum(1)[None]                                         # (1,B)
            ade_imp = (ek * free[None]).sum(-1) / nfree                       # (K,B) over free points only
            ade_ref = (e0 * free[None]).sum(-1) / nfree                       # same free points, nothing imposed
            fde_imp, fde_ref = ek[:, :, -1], e0[:, :, -1]
            a = acc[k]
            a["ade_imp"].append(ade_imp.min(0).values.cpu());  a["ade_ref"].append(ade_ref.min(0).values.cpu())
            a["fde_imp"].append(fde_imp.min(0).values.cpu());  a["fde_ref"].append(fde_ref.min(0).values.cpu())
            a["mean_ade_imp"].append(ade_imp.mean(0).cpu());   a["mean_ade_ref"].append(ade_ref.mean(0).cpu())
            a["mean_fde_imp"].append(fde_imp.mean(0).cpu());   a["mean_fde_ref"].append(fde_ref.mean(0).cpu())

        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"    batch {i+1}/{n_b}  {el/60:.1f} min elapsed, ~{el/(i+1)*(n_b-i-1)/60:.1f} min left", flush=True)

    return {k: {n: torch.cat(v).numpy() for n, v in d.items()} for k, d in acc.items()}


def print_tables(out):
    files = sorted(Path(out).expanduser().glob("summary_impose_K*.json"), key=lambda f: json.load(open(f))["K"])
    if not files:
        sys.exit(f"no summary_impose_K*.json in {out}")
    for f in files:
        s = json.load(open(f))
        print(f"\n### RoPE-TS-Impose, K={s['K']} (best of {s['K']}), test split, epoch {s['epoch']}")
        print("minADE is over the FREE points only; 'nothing imposed' = same model, same free points, no reveal. "
              "Delta = imposed - nothing imposed (m, negative = revealing helped)\n")
        print("| k imposed | minADE imposed | minADE nothing imposed | delta | minFDE imposed | minFDE nothing imposed | delta |")
        print("|---|---|---|---|---|---|---|")
        for k, r in sorted(s["ks"].items(), key=lambda kv: int(kv[0])):
            print(f"| {k} | {r['minADE_imp']:.1f} | {r['minADE_ref']:.1f} | {r['minADE_imp']-r['minADE_ref']:+.1f} | "
                  f"{r['minFDE_imp']:.1f} | {r['minFDE_ref']:.1f} | {r['minFDE_imp']-r['minFDE_ref']:+.1f} |")
        print("\nsingle-sample (mean over the K samples):\n| k | meanADE imposed | meanADE nothing imposed | meanFDE imposed | meanFDE nothing imposed |\n|---|---|---|---|---|")
        for k, r in sorted(s["ks"].items(), key=lambda kv: int(kv[0])):
            print(f"| {k} | {r['meanADE_imp']:.1f} | {r['meanADE_ref']:.1f} | {r['meanFDE_imp']:.1f} | {r['meanFDE_ref']:.1f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc_path")
    ap.add_argument("--Ks", default="5,20")
    ap.add_argument("--ks", default="0,1,2,3")
    ap.add_argument("--max_k", type=int, default=3)
    ap.add_argument("--ckpt_root", default="~")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max_batches", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_results_newSplit")
    ap.add_argument("--table", action="store_true", help="just print the finished tables from --out")
    a = ap.parse_args()

    if a.table:
        return print_tables(a.out)
    if not a.nc_path:
        sys.exit("--nc_path is required")

    import netCDF4 as nc
    from DataLoaders.ADSBdatasetBase import get_dataloaders

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "(CPU - will be very slow)")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(a.nc_path, "r"); feat_std = np.array(ds.feature_std); ds.close()
    std = torch.tensor(feat_std[:2], device=device, dtype=torch.float32)
    _, _, te = get_dataloaders(a.nc_path, batch_size=a.batch_size)
    print(f"test: {len(te.dataset)} windows")

    hits = sorted(Path(a.ckpt_root).expanduser().glob(PATTERN))
    if not hits:
        sys.exit(f"NO checkpoint matches {a.ckpt_root}/{PATTERN}")
    print("loading", hits[0], flush=True)
    model, meta = load_model(hits[0], device)
    ks = [int(k) for k in a.ks.split(",")]

    for K in [int(k) for k in a.Ks.split(",")]:
        t0 = time.time()
        res = evaluate(model, te, std, device, K, ks, a.steps, a.max_batches, a.seed, a.max_k)
        summ = {"K": K, "ckpt": str(hits[0]), **meta, "ks": {}, "minutes": (time.time() - t0) / 60}
        for k, r in res.items():
            summ["ks"][str(k)] = {
                "minADE_imp": float(r["ade_imp"].mean()), "minADE_ref": float(r["ade_ref"].mean()),
                "minFDE_imp": float(r["fde_imp"].mean()), "minFDE_ref": float(r["fde_ref"].mean()),
                "minFDE_imp_median": float(np.median(r["fde_imp"])), "minFDE_ref_median": float(np.median(r["fde_ref"])),
                "meanADE_imp": float(r["mean_ade_imp"].mean()), "meanADE_ref": float(r["mean_ade_ref"].mean()),
                "meanFDE_imp": float(r["mean_fde_imp"].mean()), "meanFDE_ref": float(r["mean_fde_ref"].mean()),
                "n": int(len(r["fde_imp"])),
            }
        np.savez(out / f"impose_K{K}.npz", **{f"k{k}_{n}": v for k, r in res.items() for n, v in r.items()})
        json.dump(summ, open(out / f"summary_impose_K{K}.json", "w"), indent=2)    # saved before the next K starts
        print(f"[K={K}] DONE in {summ['minutes']:.1f} min", flush=True)
    print_tables(out)


if __name__ == "__main__":
    main()