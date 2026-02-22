# debug_val_batch_torch.py
import argparse, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- helpers ----------
def idx_to_name(idx, midi_min):
    midi = midi_min + idx
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    return f"{names[midi % 12]}{(midi // 12) - 1}"

def format_multihot(y, midi_min, thr=0.5, max_items=6):
    if isinstance(y, torch.Tensor):
        idx = (y > thr).nonzero(as_tuple=True)[0].tolist()
    else:
        idx = np.where(y > thr)[0].tolist()
    idx = idx[:max_items]
    return " ".join(idx_to_name(i, midi_min) for i in idx) if idx else "(none)"

def load_npz_shard(dataset_dir, split, shard_idx):
    p = Path(dataset_dir) / split / f"shard_{shard_idx:03d}.npz"
    if not p.exists():
        raise FileNotFoundError(f"Shard not found: {p}")
    d = np.load(p)
    return d["X"], d["YA"], (d["YO"] if "YO" in d.files else None)

# ---------- minimal TCN-like fallback ----------
# If your checkpoint is a pure state_dict and we don't know your exact model class,
# we need SOME model definition to load into. This matches the common pattern used in
# your project: (B, bands, ctx) -> conv stack -> (B, n_notes).
class TinyTCN(nn.Module):
    def __init__(self, bands=48, ctx=16, n_notes=49, hidden=128, depth=4, k=3):
        super().__init__()
        # treat input as (B, 1, bands, ctx)
        self.conv_in = nn.Conv2d(1, hidden, kernel_size=(3,3), padding=1)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.Conv2d(hidden, hidden, kernel_size=(k,k), padding=k//2, groups=1),
                nn.GELU(),
                nn.Conv2d(hidden, hidden, kernel_size=1),
                nn.GELU(),
            ))
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(hidden, n_notes),
        )

    def forward(self, x):
        # x: (B, bands, ctx)
        x = x.unsqueeze(1)  # (B,1,bands,ctx)
        x = self.conv_in(x)
        for blk in self.blocks:
            x = x + blk(x)
        return self.head(x)

def try_build_model_from_state_dict(sd, bands, ctx, n_notes):
    """
    Best-effort: try loading into TinyTCN. If keys don't match, we print a helpful error.
    """
    model = TinyTCN(bands=bands, ctx=ctx, n_notes=n_notes)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # If it's totally incompatible, strict=False will still "load nothing" but we can detect.
    loaded_any = (len(missing) < len(model.state_dict()))
    return model, missing, unexpected, loaded_any

def load_model_any(ckpt_path, bands, ctx, n_notes):
    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Case A: full model serialized
    if isinstance(ckpt, dict) and "model" in ckpt and hasattr(ckpt["model"], "forward"):
        model = ckpt["model"].eval()
        return model, "full_model"

    # Case B: common dict formats
    state_dict = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif all(isinstance(k, str) for k in ckpt.keys()):
            # might itself be a raw state_dict
            state_dict = ckpt

    if state_dict is None:
        raise RuntimeError(f"Don't know how to load checkpoint type={type(ckpt)} keys={getattr(ckpt,'keys',lambda:None)()}")

    model, missing, unexpected, loaded_any = try_build_model_from_state_dict(state_dict, bands, ctx, n_notes)
    if not loaded_any:
        raise RuntimeError(
            "Checkpoint looks like a state_dict, but it doesn't match TinyTCN.\n"
            "Paste the output of:\n"
            "  python -c \"import torch; c=torch.load('fb_model_torch/best.pt',map_location='cpu'); "
            "print(type(c)); print(c.keys() if hasattr(c,'keys') else None); "
            "sd=c.get('state_dict',c.get('model_state_dict',c)); "
            "print('sd keys sample:', list(sd.keys())[:20])\"\n"
            "…and I’ll generate the exact matching model class file for your project."
        )
    return model.eval(), f"state_dict (missing={len(missing)} unexpected={len(unexpected)})"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset_dir", required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--thr", type=float, default=0.5)
    args = ap.parse_args()

    meta = json.load(open(Path(args.dataset_dir) / "metadata.json", "r"))
    midi_min = meta["label"]["midi_min"]
    bands = meta["bands"]
    ctx = meta["ctx_frames"]
    n_notes = meta["label"]["n_notes"]

    X, YA, YO = load_npz_shard(args.dataset_dir, args.split, args.shard)

    # dataset X is float16; cast for torch
    Xb = torch.from_numpy(X[:args.batch]).float()         # (B, 48, 16)
    YAb = torch.from_numpy(YA[:args.batch]).float()       # (B, 49)

    model, how = load_model_any(args.ckpt, bands=bands, ctx=ctx, n_notes=n_notes)
    print("Loaded model:", how)

    with torch.no_grad():
        logits = model(Xb)
        probs = torch.sigmoid(logits)

    for i in range(args.batch):
        pred = format_multihot(probs[i], midi_min, thr=args.thr, max_items=6)
        true = format_multihot(YAb[i], midi_min, thr=0.5, max_items=6)

        topv, topi = torch.topk(probs[i], k=6)
        top = " ".join(f"{idx_to_name(int(j), midi_min)}({float(v):.2f})" for v, j in zip(topv, topi))
        print(f"{i:03d}  pred={pred:<30}  true={true:<30}  top6={top}")

if __name__ == "__main__":
    main()