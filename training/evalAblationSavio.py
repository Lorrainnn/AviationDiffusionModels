"""
evalAblationSavio.py - evaluate best checkpoints at K=5 then K=20, saving after EVERY (K, model).

K=5 is run first for the whole requested model list, so those numbers exist (and are on disk)
long before K=20 finishes. Each finished (model, K) writes its own files, no shared file, so
several array tasks can run in parallel without clobbering each other:

    <out>/summary_<model>_K<K>.json        numbers for that model at that K (+ hold/CV baselines)
    <out>/test_<model>_best_K<K>.npz       per-trajectory errors

Run from training/ on Savio (the sbatch file does this):
    python -u evalAblationSavio.py --nc_path $NC_PATH --models RoPE-TS --Ks 5,20 --out $HOME/eval_results_newSplit
"""
import argparse, importlib, json, sys, time
from pathlib import Path

import numpy as np
import torch
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).parent.parent))
from DataLoaders.ADSBdatasetBase import get_dataloaders   # mentor's flight-level split
from models.cfm import euler_sample

# glob patterns are resolved under --ckpt_root (default: your home dir on Savio)
RUNS = {
    "CFM-NewSplit":   ("models.dit",                  "checkpoints_cfm_newSplit*/best*.pt"),
    "RoPE-OG":        ("models.dit_RoPE_original",    "checkpoints_cfm_rope_og_newSplit*/best*.pt"),
    "RoPE-TS":        ("models.dit_RoPE_timestamps",  "checkpoints_cfm_rope_ts_newSplit*/best*.pt"),
    "RoPE-TS-Impose": ("models.ditRoPETS_impose",     "checkpoints_cfm_ts_impose_newSplit*/best*.pt"),
}


def resolve(root, pattern):
    hits = sorted(Path(root).expanduser().glob(pattern))
    return hits[0] if hits else None


def load_model(module_name, ckpt_path, device):
    mod = importlib.import_module(module_name)
    model = mod.TrajectoryDiT(d_model=256, n_heads=8, n_layers=6).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["ema_state"])        # EMA weights, same as validate()
    model.eval()
    meta = {
        "epoch": int(ckpt["epoch"]) + 1,             # saved epoch is 0-indexed
        "logged_val_fde": ckpt.get("val_fde"),
        "params": sum(p.numel() for p in model.parameters()),
    }
    return model, meta


@torch.no_grad()
def evaluate(model, loader, std, device, K, steps, max_batches, seed):
    torch.manual_seed(seed)
    fde_min, fde_mean, ade_min, ade_mean, fde_one = [], [], [], [], []
    hor_min, hor_mean = [], []
    t0 = time.time()
    n_b = len(loader)
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs, fut, trel = b["obs"].to(device), b["fut"].to(device), b["t_rel"].to(device)
        preds = euler_sample(model, obs, trel, n_samples=K, n_steps=steps, device=str(device))
        err = torch.norm(preds[..., :2] * std - (fut[..., :2] * std)[None], dim=-1)  # (K,B,43)
        fde, ade = err[:, :, -1], err.mean(-1)                                       # (K,B)
        fde_one.append(fde[0].cpu())
        fde_min.append(fde.min(0).values.cpu());  fde_mean.append(fde.mean(0).cpu())
        ade_min.append(ade.min(0).values.cpu());  ade_mean.append(ade.mean(0).cpu())
        hor_min.append(err.min(0).values.cpu());  hor_mean.append(err.mean(0).cpu())
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"    batch {i+1}/{n_b}  {el/60:.1f} min elapsed, ~{el/(i+1)*(n_b-i-1)/60:.1f} min left", flush=True)
    cat = lambda xs: torch.cat(xs).numpy()
    return dict(FDE1=cat(fde_one), minFDE=cat(fde_min), meanFDE=cat(fde_mean), minADE=cat(ade_min),
                meanADE=cat(ade_mean), hor_min=cat(hor_min), hor_mean=cat(hor_mean))


@torch.no_grad()
def baselines(loader, std, device, max_batches, k=5):
    """Hold-last-position and constant-velocity (from last k obs points) FDE/ADE."""
    out = {"hold": [], "cv": []}
    for i, b in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        obs, fut, trel = b["obs"].to(device), b["fut"].to(device), b["t_rel"].to(device)
        p, g = obs[..., :2] * std, fut[..., :2] * std
        t_obs, t_fut = trel[:, :43], trel[:, 43:]
        last = p[:, -1]
        hold = last[:, None].expand_as(g)
        dt = (t_obs[:, -1] - t_obs[:, -k]).clamp(min=1e-6)
        v = (p[:, -1] - p[:, -k]) / dt[:, None]
        cv = last[:, None] + v[:, None] * (t_fut - t_obs[:, -1:])[..., None]
        for name, pr in (("hold", hold), ("cv", cv)):
            e = torch.norm(pr - g, dim=-1)
            out[name].append(torch.stack([e[:, -1], e.mean(-1)], 1).cpu())
    return {n: torch.cat(v).numpy() for n, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc_path", required=True)
    ap.add_argument("--models", default=",".join(RUNS), help="comma list of: " + ",".join(RUNS))
    ap.add_argument("--Ks", default="5,20", help="run in this order; each K is saved before the next starts")
    ap.add_argument("--ckpt_root", default="~")
    ap.add_argument("--split", default="test")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--max_batches", type=int, default=0, help="0 = full split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="eval_results_newSplit")
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "(CPU - will be very slow)")
    out = Path(a.out).expanduser(); out.mkdir(parents=True, exist_ok=True)

    ds = nc.Dataset(a.nc_path, "r"); feat_std = np.array(ds.feature_std); ds.close()
    std = torch.tensor(feat_std[:2], device=device, dtype=torch.float32)

    tr, va, te = get_dataloaders(a.nc_path, batch_size=a.batch_size)
    loader = {"val": va, "test": te}[a.split]
    print(f"{a.split}: {len(loader.dataset)} windows")

    bl = baselines(loader, std, device, a.max_batches)
    bl = {n: {"FDE": float(x[:, 0].mean()), "ADE": float(x[:, 1].mean())} for n, x in bl.items()}
    print("baselines:", bl)

    names = [n.strip() for n in a.models.split(",")]
    for K in [int(k) for k in a.Ks.split(",")]:
        for name in names:
            mod, pattern = RUNS[name]
            path = resolve(a.ckpt_root, pattern)
            if path is None:
                print(f"[K={K}] {name}: NO checkpoint matches {a.ckpt_root}/{pattern}, skipping", flush=True)
                continue
            print(f"[K={K}] {name}: loading {path}", flush=True)
            model, meta = load_model(mod, path, device)
            t0 = time.time()
            r = evaluate(model, loader, std, device, K, a.steps, a.max_batches, a.seed)
            f = r["minFDE"]
            s = {
                "model": name, "K": K, "split": a.split, **meta, "ckpt": str(path),
                "minFDE": float(f.mean()), "minFDE_median": float(np.median(f)),
                "minFDE_p90": float(np.percentile(f, 90)), "minFDE_p99": float(np.percentile(f, 99)),
                "minADE": float(r["minADE"].mean()),
                "meanFDE": float(r["meanFDE"].mean()), "meanADE": float(r["meanADE"].mean()),
                "FDE1": float(r["FDE1"].mean()), "FDE1_median": float(np.median(r["FDE1"])),
                "FDE1_p90": float(np.percentile(r["FDE1"], 90)), "FDE1_p99": float(np.percentile(r["FDE1"], 99)),
                "n": int(len(f)), "minutes": (time.time() - t0) / 60, "baselines": bl,
            }
            np.savez(out / f"{a.split}_{name}_best_K{K}.npz", **r)           # saved immediately
            json.dump(s, open(out / f"summary_{name}_K{K}.json", "w"), indent=2)
            print(f"[K={K}] {name} DONE in {s['minutes']:.1f} min  minADE {s['minADE']:.1f}  "
                  f"minFDE {s['minFDE']:.1f}  meanADE {s['meanADE']:.1f}  meanFDE {s['meanFDE']:.1f}", flush=True)
            del model; torch.cuda.empty_cache()


if __name__ == "__main__":
    main()