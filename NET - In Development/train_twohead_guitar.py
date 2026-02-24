"""
train_twohead_guitar.py

Stage 1 (default): single-note transient classification
  - Loss: Softmax Cross Entropy (nn.CrossEntropyLoss)
  - Label: single class index (0..V-1) derived from multi_hot argmax
  - Metrics: top1 / top3 / mean_true_p

Stage 2 (--stage2): multi-label (single + chords)
  - Loss: BCEWithLogitsLoss
  - Label: multi-hot (V,)
  - Metrics: best-F1 over threshold sweep + fixed-thresh F1

Visualizer:
  - Wiring-style network diagram with animated pulse path (inputs -> conv layers -> outputs)
  - Shows current input waveform (10–15ms) and top predictions
  - NOW draws a vertical line showing the onset sample index inside the *returned* window
  - Also prints the current true input note string

Context windowing:
  - base window length: --train_ms / --val_ms (e.g. 15ms = 720 samples @ 48k)
  - optional cropping to inference window: --crop_ms / --val_crop_ms (e.g. 10ms = 480 samples)
  - crop probability: --crop_prob / --val_crop_prob
  - onset anchoring inside base window: --onset_pos_min / --onset_pos_max
  - forcing crop to include onset: --crop_keep_onset_prob

IMPORTANT:
  - Dataset returns (x, y, onset_idx_in_returned_window)
  - DataLoader uses a padding collate_fn so mixed 480/720 batches stack fine

COMMAND: 
    python train_twohead_guitar.py   --dataset labels   --out stage1_ce   --epochs 80   --width 64   --p_on 1.0 --p_neg 0.0   --val_p_on 1.0 --val_p_neg 0.0   --scheduler step --step_size 25 --step_gamma 0.5   --curriculum_epochs 15   --calm_noise_std 0.0 --calm_enable_polarity 0 --calm_gain_min -4 --calm_gain_max 2   --full_noise_std 0.0005 --full_enable_polarity 1 --full_gain_min -18 --full_gain_max 6   --train_ms 15 --crop_ms 10 --crop_prob 1.0   --val_ms 15 --val_crop_ms 10 --val_crop_prob 1.0   --crop_keep_onset_prob 1.0   --viz --viz_every 10 --viz_weight_every 10 --viz_pulse_speed 0.5 
"""

import os
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional, Union

import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler


# ----------------------------
# Constants
# ----------------------------
DEFAULT_INFER_MS = 10.0  # Daisy target (10ms @ 48kHz = 480 samples)


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def read_manifest(path: str) -> List[Dict[str, str]]:
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def ms_to_samples(ms: float, sr: int) -> int:
    return int(round((ms / 1000.0) * float(sr)))


def label_to_note_name_multi(label_vec: torch.Tensor, meta: Dict[str, Any]) -> str:
    pitches = meta.get("index_to_pitch", None)
    idxs = (label_vec >= 0.5).nonzero(as_tuple=False).flatten().tolist()
    if not idxs:
        return "(none)"
    if pitches is None:
        return ",".join(str(i) for i in idxs)
    names = [pitches[i] if i < len(pitches) else str(i) for i in idxs]
    return ",".join(names)


def label_to_note_name_single(class_idx: int, meta: Dict[str, Any]) -> str:
    pitches = meta.get("index_to_pitch", None)
    if pitches is None:
        return str(class_idx)
    if 0 <= class_idx < len(pitches):
        return pitches[class_idx]
    return str(class_idx)


def f1_scores_from_logits(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5):
    eps = 1e-9
    probs = torch.sigmoid(logits)
    preds = (probs >= thresh).float()
    tp = (preds * targets).sum().item()
    fp = (preds * (1.0 - targets)).sum().item()
    fn = ((1.0 - preds) * targets).sum().item()
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1   = 2.0 * prec * rec / (prec + rec + eps)
    return float(prec), float(rec), float(f1)


def best_f1_over_thresholds(logits: torch.Tensor, targets: torch.Tensor, thresholds: List[float]):
    best = (-1.0, 0.5, 0.0, 0.0)  # (f1, thresh, p, r)
    for th in thresholds:
        p, r, f1 = f1_scores_from_logits(logits, targets, thresh=th)
        if f1 > best[0]:
            best = (f1, float(th), p, r)
    return best  # f1, thresh, p, r


def maybe_linear_warmup_lr(opt: torch.optim.Optimizer, base_lr: float, step_idx: int, warmup_steps: int):
    if warmup_steps <= 0:
        return
    scale = min(1.0, (step_idx + 1) / float(warmup_steps))
    lr_now = base_lr * scale
    for pg in opt.param_groups:
        pg["lr"] = lr_now


def pad_collate_1d_with_onset(batch):
    """
    Pads variable-length audio tensors in a batch to the max length.

    Each item: (x, y, onset_idx)
      x: (1, T) float tensor
      y: stage1 -> scalar long tensor, stage2 -> (V,) float tensor
      onset_idx: int (index in returned x), -1 if unknown/neg

    Returns:
      x_padded: (B, 1, Tmax)
      y_stacked: (B,) or (B, V)
      onset_idx: (B,) long tensor (still refers to *pre-pad* indices; padding is only appended)
    """
    xs, ys, ons = zip(*batch)
    max_T = max(int(x.shape[-1]) for x in xs)

    x_out = []
    for x in xs:
        T = int(x.shape[-1])
        if T < max_T:
            pad = torch.zeros((x.shape[0], max_T - T), dtype=x.dtype)
            x = torch.cat([x, pad], dim=-1)
        x_out.append(x)
    x_out = torch.stack(x_out, dim=0)

    y_out = torch.stack(ys, dim=0)
    on_out = torch.tensor(ons, dtype=torch.long)
    return x_out, y_out, on_out


# ----------------------------
# Live Net Visualizer (optional)
# ----------------------------
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


class LiveNetViz:
    """
    Wiring-diagram style visualizer; works for both CE stage and BCE stage.

    NEW:
      - Draws a vertical onset line in the waveform plot using onset_idx provided by dataset.
    """
    def __init__(
        self,
        model: nn.Module,
        meta: Dict[str, Any],
        sr: int,
        thresh: float = 0.5,
        topk: int = 8,
        out_nodes_cap: int = 56,
        weight_viz_every: int = 3,
        pulse_speed: float = 0.45,
    ):
        self.model = model
        self.meta = meta or {}
        self.sr = int(sr)
        self.thresh = float(thresh)
        self.topk = int(topk)
        self.pitches = self.meta.get("index_to_pitch", None)

        self.weight_viz_every = max(1, int(weight_viz_every))
        self.pulse_speed = float(pulse_speed)

        self.conv0 = model.conv[0]
        self.conv1 = model.conv[3]
        self.conv2 = model.conv[6]
        self.head  = model.head

        self.n_in = 8
        self.n_c0 = int(self.conv0.out_channels)
        self.n_c1 = int(self.conv1.out_channels)
        self.n_c2 = int(self.conv2.out_channels)
        self.n_out_full = int(self.head.out_features)
        self.n_out = min(out_nodes_cap, self.n_out_full)

        self.act_c0 = None
        self.act_c1 = None
        self.act_c2 = None
        self._install_hooks()

        self._viz_tick = 0
        self._pulse_pos = 0.0
        self._last_path: Optional[Dict[str, int]] = None

        plt.ion()
        self.fig = plt.figure(figsize=(14, 6), facecolor="black")
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1, 1])

        self.ax_net  = self.fig.add_subplot(gs[:, 0])
        self.ax_wave = self.fig.add_subplot(gs[0, 1])
        self.ax_pred = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_net, self.ax_wave, self.ax_pred):
            ax.set_facecolor("black")
        self.ax_net.set_axis_off()

        self.pos = self._make_positions()
        self.node_artists: Dict[Tuple[str, int], Circle] = {}
        self.edge_artists: List[Tuple[str, str, Tuple[str, int], Tuple[str, int], Line2D]] = []
        self.edge_lookup: Dict[Tuple[str, str, int, int], Line2D] = {}
        self.edge_base_style: Dict[Line2D, Tuple[Tuple[float, float, float], float, float]] = {}

        self._build_static_artists()
        self.fig.tight_layout(pad=1.0)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def close(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass

    def _install_hooks(self):
        def hook_c0(_, __, out): self.act_c0 = out.detach()
        def hook_c1(_, __, out): self.act_c1 = out.detach()
        def hook_c2(_, __, out): self.act_c2 = out.detach()

        self.conv0.register_forward_hook(hook_c0)
        self.conv1.register_forward_hook(hook_c1)
        self.conv2.register_forward_hook(hook_c2)

    def _make_positions(self):
        pos: Dict[Tuple[str, int], Tuple[float, float]] = {}
        cols = [("in", self.n_in), ("c0", self.n_c0), ("c1", self.n_c1), ("c2", self.n_c2), ("out", self.n_out)]
        xs = [0.05, 0.30, 0.52, 0.74, 0.95]
        for (name, n), x in zip(cols, xs):
            ys = [0.5] if n <= 1 else list(reversed([0.1 + 0.8 * (i/(n-1)) for i in range(n)]))
            for i, y in enumerate(ys):
                pos[(name, i)] = (x, y)
        return pos

    def _build_static_artists(self):
        def add_node(key, radius=0.018):
            x, y = self.pos[key]
            circ = Circle((x, y), radius=radius, facecolor=(0, 0, 0, 1), edgecolor=(1, 1, 1, 1), linewidth=2.0)
            self.ax_net.add_patch(circ)
            self.node_artists[key] = circ

        for i in range(self.n_in): add_node(("in", i), radius=0.020)
        for i in range(self.n_c0): add_node(("c0", i), radius=0.018)
        for i in range(self.n_c1): add_node(("c1", i), radius=0.018)
        for i in range(self.n_c2): add_node(("c2", i), radius=0.018)
        for i in range(self.n_out): add_node(("out", i), radius=0.016)

        self._add_dense_edges("in", "c0")
        self._add_dense_edges("c0", "c1")
        self._add_dense_edges("c1", "c2")
        self._add_dense_edges("c2", "out")

    def _add_dense_edges(self, a: str, b: str):
        keys_a = [k for k in self.pos.keys() if k[0] == a]
        keys_b = [k for k in self.pos.keys() if k[0] == b]
        for ka in keys_a:
            xa, ya = self.pos[ka]
            for kb in keys_b:
                xb, yb = self.pos[kb]
                line = Line2D([xa, xb], [ya, yb], linewidth=0.5, alpha=0.08, color=(0.35, 0.35, 0.35))
                self.ax_net.add_line(line)
                self.edge_artists.append((a, b, ka, kb, line))
                self.edge_lookup[(a, b, ka[1], kb[1])] = line

    @staticmethod
    def _color_from_weight(w: float):
        return (0.35, 1.00, 0.35) if w >= 0 else (1.00, 0.35, 0.35)

    def _get_weight_mats(self):
        w0 = self.conv0.weight.detach().float().mean(dim=2).squeeze(1)  # (C0,)
        w1 = self.conv1.weight.detach().float().mean(dim=2)             # (C1,C0)
        w2 = self.conv2.weight.detach().float().mean(dim=2)             # (C2,C1)
        wh = self.head.weight.detach().float()[: self.n_out]            # (n_out,C2)
        return w0, w1, w2, wh

    def _get_act_vecs(self):
        def chan_mag(t):
            if t is None:
                return None
            return t.detach().float().abs().mean(dim=(0, 2)).cpu()
        return chan_mag(self.act_c0), chan_mag(self.act_c1), chan_mag(self.act_c2)

    def _update_nodes_glow(self):
        a0, a1, a2 = self._get_act_vecs()
        def set_glow(layer, mags):
            if mags is None:
                return
            mags = mags / (mags.max().clamp(min=1e-6))
            for i in range(len(mags)):
                key = (layer, i)
                if key in self.node_artists:
                    g = float(mags[i].item())
                    self.node_artists[key].set_facecolor((0.10, 0.60, 0.10, 0.06 + 0.55 * g))
        set_glow("c0", a0)
        set_glow("c1", a1)
        set_glow("c2", a2)

    def _update_edges_weights(self):
        w0, w1, w2, wh = self._get_weight_mats()
        def norm_abs(x):
            x = x.abs()
            return x / (x.max().clamp(min=1e-6))
        w0n, w1n, w2n, whn = norm_abs(w0.cpu()), norm_abs(w1.cpu()), norm_abs(w2.cpu()), norm_abs(wh.cpu())
        for a, b, ka, kb, line in self.edge_artists:
            ia, ib = ka[1], kb[1]
            if a == "in" and b == "c0":
                w, s = float(w0[ib].cpu().item()), float(w0n[ib].item())
            elif a == "c0" and b == "c1":
                w, s = float(w1[ib, ia].cpu().item()), float(w1n[ib, ia].item())
            elif a == "c1" and b == "c2":
                w, s = float(w2[ib, ia].cpu().item()), float(w2n[ib, ia].item())
            elif a == "c2" and b == "out":
                w, s = float(wh[ib, ia].cpu().item()), float(whn[ib, ia].item())
            else:
                continue
            col = self._color_from_weight(w)
            lw = 0.12 + 2.2 * s
            al = 0.02 + 0.55 * s
            line.set_color(col)
            line.set_linewidth(lw)
            line.set_alpha(al)
            self.edge_base_style[line] = (col, lw, al)

    def _choose_input_bucket(self, x_np: np.ndarray) -> int:
        if x_np.size == 0:
            return 0
        segs = np.array_split(x_np, self.n_in)
        energies = np.array([float(np.mean(s * s)) if s.size else 0.0 for s in segs], dtype=np.float32)
        return int(np.argmax(energies))

    def _choose_path(self, x_1: torch.Tensor, logits_1: torch.Tensor) -> Dict[str, int]:
        x_np = x_1.detach().float().cpu().numpy().reshape(-1)
        in_idx = self._choose_input_bucket(x_np)
        a0, a1, a2 = self._get_act_vecs()
        c0 = int(torch.argmax(a0).item()) if a0 is not None else 0
        c1 = int(torch.argmax(a1).item()) if a1 is not None else 0
        c2 = int(torch.argmax(a2).item()) if a2 is not None else 0
        out_full = int(torch.argmax(torch.sigmoid(logits_1)).item())
        out = min(out_full, self.n_out - 1)
        return {"in": in_idx, "c0": c0, "c1": c1, "c2": c2, "out": out}

    def _revert_last_pulse(self):
        if self._last_path is None:
            return
        p = self._last_path
        segs = [("in","c0",p["in"],p["c0"]), ("c0","c1",p["c0"],p["c1"]),
                ("c1","c2",p["c1"],p["c2"]), ("c2","out",p["c2"],p["out"])]
        for a,b,si,di in segs:
            line = self.edge_lookup.get((a,b,si,di))
            if line is None:
                continue
            base = self.edge_base_style.get(line)
            if base is None:
                continue
            col,lw,al = base
            line.set_color(col); line.set_linewidth(lw); line.set_alpha(al)

    def _apply_pulse(self, path: Dict[str, int]):
        self._revert_last_pulse()
        self._pulse_pos += self.pulse_speed
        if self._pulse_pos >= 4.0:
            self._pulse_pos = 0.0
        segs = [("in","c0",path["in"],path["c0"]), ("c0","c1",path["c0"],path["c1"]),
                ("c1","c2",path["c1"],path["c2"]), ("c2","out",path["c2"],path["out"])]
        hot = int(np.floor(self._pulse_pos))
        frac = float(self._pulse_pos - hot)
        pulse = 0.5 + 0.5 * np.sin(2.0 * np.pi * frac)

        for si,(a,b,src_i,dst_i) in enumerate(segs):
            line = self.edge_lookup.get((a,b,src_i,dst_i))
            if line is None:
                continue
            base = self.edge_base_style.get(line, ((0.35,0.35,0.35), 0.5, 0.08))
            col, lw0, al0 = base
            if si < hot:
                boost = 0.85
            elif si == hot:
                boost = 0.35 + 0.65 * pulse
            else:
                boost = 0.0
            line.set_color(col)
            line.set_linewidth(lw0 + 5.0 * boost)
            line.set_alpha(min(1.0, al0 + 0.85 * boost))

        for layer in ("in","c0","c1","c2","out"):
            idx = path[layer]
            key = (layer, idx)
            if key in self.node_artists:
                if layer == "in":
                    self.node_artists[key].set_facecolor((0.10, 0.55, 0.70, 0.85))
                elif layer == "out":
                    self.node_artists[key].set_facecolor((0.80, 0.75, 0.15, 0.85))
                else:
                    self.node_artists[key].set_facecolor((0.15, 0.75, 0.20, 0.85))
        self._last_path = dict(path)

    def _update_wave_and_preds(self, x_1: torch.Tensor, logits_1: torch.Tensor, true_str: str, onset_idx: int):
        x_np = x_1.detach().float().cpu().numpy().reshape(-1)
        probs = torch.sigmoid(logits_1.detach()).cpu().numpy().reshape(-1)

        j = int(np.argmax(probs))
        p = float(probs[j])
        pred_name = self.pitches[j] if (self.pitches is not None and j < len(self.pitches)) else str(j)

        self.ax_wave.clear()
        self.ax_wave.set_facecolor("black")
        self.ax_wave.set_title(
            f"Input ({len(x_np)/self.sr*1000.0:.1f}ms) — TRUE: {true_str} | PRED: {pred_name} ({p:.2f})",
            color="white"
        )
        t = np.arange(len(x_np)) / float(self.sr) * 1000.0
        self.ax_wave.plot(t, x_np, linewidth=1.0)

        # Draw onset marker if known
        if onset_idx is not None and int(onset_idx) >= 0 and int(onset_idx) < len(x_np):
            onset_ms = (float(onset_idx) / float(self.sr)) * 1000.0
            self.ax_wave.axvline(onset_ms, linewidth=2.0)

        self.ax_wave.set_xlim(0, t[-1] if len(t) else 1.0)
        self.ax_wave.tick_params(colors="white")
        for spine in self.ax_wave.spines.values():
            spine.set_color("white")

        self.ax_pred.clear()
        self.ax_pred.set_facecolor("black")
        self.ax_pred.set_title("Top predictions (sigmoid view)", color="white")
        topk = min(self.topk, len(probs))
        idxs = np.argsort(-probs)[:topk]
        vals = probs[idxs]
        labels = [(self.pitches[k] if (self.pitches is not None and k < len(self.pitches)) else str(k)) for k in idxs]
        self.ax_pred.barh(range(topk)[::-1], vals[::-1])
        self.ax_pred.set_yticks(range(topk)[::-1])
        self.ax_pred.set_yticklabels(labels[::-1], color="white")
        self.ax_pred.set_xlim(0.0, 1.0)
        self.ax_pred.axvline(self.thresh, linestyle="--", linewidth=1.0)
        self.ax_pred.tick_params(colors="white")
        for spine in self.ax_pred.spines.values():
            spine.set_color("white")

    @torch.no_grad()
    def update(self, x_batch: torch.Tensor, y_info: torch.Tensor, onset_idx: torch.Tensor, device: str, stage2: bool):
        self.model.eval()
        x = x_batch[:1].to(device, non_blocking=True)
        logits = self.model(x)

        # onset index for this displayed sample (still valid after pad because padding is appended)
        oi = int(onset_idx[:1].detach().cpu().item())

        if stage2:
            y = y_info[:1].detach().cpu()
            true_str = label_to_note_name_multi(y[0], self.meta)
        else:
            y = int(y_info[:1].detach().cpu().item())
            true_str = label_to_note_name_single(y, self.meta)

        self._viz_tick += 1
        if (self._viz_tick % self.weight_viz_every) == 0 or not self.edge_base_style:
            self._update_edges_weights()
        self._update_nodes_glow()

        path = self._choose_path(x_1=x[0], logits_1=logits[0])
        self._apply_pulse(path)
        self._update_wave_and_preds(x_1=x[0], logits_1=logits[0], true_str=true_str, onset_idx=oi)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)


# ----------------------------
# Data
# ----------------------------
@dataclass
class ClipInfo:
    audio_abs: str
    multi_hot: List[int]
    onset_sample: int
    transient_end: int
    num_samples: int
    sr: int
    clip_type: str  # "single", "scale_segment", "chord", "note_repeat_segment"


def load_clips(dataset_root: str) -> Tuple[List[ClipInfo], int, int, Dict[str, Any]]:
    meta = load_json(os.path.join(dataset_root, "metadata.json"))
    sr = int(meta["sr"])
    vocab_size = len(meta["midi_vocab"])

    manifest = read_manifest(os.path.join(dataset_root, "manifest.jsonl"))
    clips: List[ClipInfo] = []
    for it in manifest:
        audio_abs = os.path.join(dataset_root, it["audio"])
        lab = load_json(os.path.join(dataset_root, it["label"]))
        mh = lab["multi_hot"]
        onset = int(lab.get("onset_sample", 0))
        t_end = int(lab.get("transient_window_end", onset + int(0.005 * sr)))
        n = int(lab.get("num_samples", 0))
        clip_type = lab.get("type", it.get("type", "unknown"))
        clips.append(ClipInfo(
            audio_abs=audio_abs,
            multi_hot=mh,
            onset_sample=onset,
            transient_end=t_end,
            num_samples=n,
            sr=sr,
            clip_type=clip_type,
        ))
    return clips, sr, vocab_size, meta


def train_val_split(clips: List[ClipInfo], val_ratio: float, seed: int):
    rng = random.Random(seed)
    idxs = list(range(len(clips)))
    rng.shuffle(idxs)
    n_val = int(round(len(idxs) * val_ratio))
    val_set = set(idxs[:n_val])
    train = [c for i, c in enumerate(clips) if i not in val_set]
    val   = [c for i, c in enumerate(clips) if i in val_set]
    return train, val


class OnsetDataset(Dataset):
    """
    Each sample loads base_W samples, then optionally crops to crop_W before returning.

    NEW:
      - Returns onset_idx_in_returned_window so the viz can draw the onset marker.
      - Uses onset anchoring so the base window isn't "tail end".
      - Can force the crop to include the onset.

    Returns: (x, y, onset_idx)
      x: (1, T)
      y: stage1 -> scalar class index (Long), stage2 -> multi-hot (Float)
      onset_idx: int in [0,T), or -1 if unknown/negative
    """

    def __init__(
        self,
        clips: List[ClipInfo],
        base_W: int,
        crop_W: int,
        crop_prob: float,
        vocab_size: int,
        stage2: bool,
        p_on: float,
        p_neg: float,
        virtual_len: int = 100000,
        audio_cache_max: int = 256,
        preemph_coef: float = 0.0,
        # Aug
        noise_std: float = 0.0,
        gain_db_min: float = -12.0,
        gain_db_max: float = 3.0,
        enable_gain: bool = True,
        enable_polarity: bool = False,
        # Onset anchoring / onset-aware crop (positives only)
        onset_pos_min: float = 0.20,
        onset_pos_max: float = 0.65,
        crop_keep_onset_prob: float = 0.80,
    ):
        self.clips = clips
        self.base_W = int(base_W)
        self.crop_W = int(crop_W)
        self.crop_prob = float(crop_prob)
        self.V = int(vocab_size)
        self.stage2 = bool(stage2)
        self.virtual_len = int(virtual_len)

        self.preemph_coef = float(preemph_coef)

        # Aug profile (mutable via set_aug_profile)
        self.noise_std = float(noise_std)
        self.gain_db_min = float(gain_db_min)
        self.gain_db_max = float(gain_db_max)
        self.enable_gain = bool(enable_gain)
        self.enable_polarity = bool(enable_polarity)

        # Onset anchor config
        self.onset_pos_min = float(onset_pos_min)
        self.onset_pos_max = float(onset_pos_max)
        self.crop_keep_onset_prob = float(np.clip(crop_keep_onset_prob, 0.0, 1.0))

        s = p_on + p_neg
        self.p_on  = p_on / s
        self.p_neg = p_neg / s

        self.on_candidates = [i for i, c in enumerate(clips) if c.transient_end > c.onset_sample]

        self.audio_cache: Dict[str, np.ndarray] = {}
        self.cache_order: List[str] = []
        self.cache_max = int(audio_cache_max)

        # Sanity for crop sizes
        if self.crop_W <= 0:
            self.crop_W = self.base_W
        if self.crop_W > self.base_W:
            self.crop_W = self.base_W
        self.crop_prob = float(np.clip(self.crop_prob, 0.0, 1.0))

    def set_aug_profile(
        self,
        *,
        noise_std: float,
        gain_db_min: float,
        gain_db_max: float,
        enable_gain: bool,
        enable_polarity: bool,
    ):
        self.noise_std = float(noise_std)
        self.gain_db_min = float(gain_db_min)
        self.gain_db_max = float(gain_db_max)
        self.enable_gain = bool(enable_gain)
        self.enable_polarity = bool(enable_polarity)

    def __len__(self):
        return self.virtual_len

    def _load(self, path: str, sr_expected: int) -> np.ndarray:
        if path in self.audio_cache:
            return self.audio_cache[path]
        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        assert int(sr) == int(sr_expected), f"SR mismatch {path}"
        self.audio_cache[path] = y
        self.cache_order.append(path)
        if len(self.cache_order) > self.cache_max:
            old = self.cache_order.pop(0)
            self.audio_cache.pop(old, None)
        return y

    def _augment(self, x: np.ndarray) -> np.ndarray:
        if self.enable_gain:
            gain_db = random.uniform(self.gain_db_min, self.gain_db_max)
            x = x * (10.0 ** (gain_db / 20.0))
        x = np.clip(x, -1.0, 1.0)

        if self.noise_std > 0:
            x = x + np.random.randn(len(x)).astype(np.float32) * self.noise_std

        if self.enable_polarity and random.random() < 0.5:
            x = -x

        return x

    def _rms_norm(self, x: np.ndarray) -> np.ndarray:
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        return x / max(rms, 1e-4)

    def _multi_hot_to_single_index(self, mh: List[int]) -> int:
        arr = np.array(mh, dtype=np.float32)
        return int(arr.argmax())

    def _maybe_crop(self, x: np.ndarray, onset_offset_in_base: Optional[int], is_pos: bool) -> Tuple[np.ndarray, int]:
        """
        Crop base window to crop_W.

        Returns:
          x_out, crop_start
        crop_start = 0 when not cropped.

        If is_pos and onset_offset_in_base is provided, we can force the crop to contain onset
        with probability crop_keep_onset_prob.
        """
        if self.crop_W >= self.base_W:
            return x, 0

        if random.random() > self.crop_prob:
            return x, 0

        max_start = self.base_W - self.crop_W

        if is_pos and (onset_offset_in_base is not None) and (random.random() < self.crop_keep_onset_prob):
            lo = max(0, onset_offset_in_base - (self.crop_W - 1))
            hi = min(max_start, onset_offset_in_base)
            st = random.randint(lo, hi) if hi >= lo else random.randint(0, max_start)
        else:
            st = random.randint(0, max_start)

        return x[st: st + self.crop_W], st

    def __getitem__(self, _):
        mode = "on" if (random.random() < self.p_on and self.on_candidates) else "neg"

        if mode == "on":
            ci = self.clips[random.choice(self.on_candidates)]
            y  = self._load(ci.audio_abs, ci.sr)

            # POSITIVE: anchor base window around onset
            min_pos = int(self.onset_pos_min * self.base_W)
            max_pos = int(self.onset_pos_max * self.base_W)
            min_pos = int(np.clip(min_pos, 0, self.base_W - 1))
            max_pos = int(np.clip(max_pos, min_pos, self.base_W - 1))

            onset_offset = random.randint(min_pos, max_pos)

            max_start = max(0, ci.num_samples - self.base_W)
            start = ci.onset_sample - onset_offset
            start = int(np.clip(start, 0, max_start))

            x = y[start: start + self.base_W]
            if len(x) < self.base_W:
                x = np.pad(x, (0, self.base_W - len(x)))

            onset_offset_in_base = int(ci.onset_sample - start)

            if self.stage2:
                label = np.array(ci.multi_hot, dtype=np.float32)
            else:
                label = self._multi_hot_to_single_index(ci.multi_hot)

            is_pos = True

        else:
            # NEGATIVE: silence before transient or after tail
            ci = self.clips[random.randrange(len(self.clips))]
            y  = self._load(ci.audio_abs, ci.sr)

            max_start = max(0, ci.num_samples - self.base_W)
            pre_end = max(0, ci.onset_sample - self.base_W)

            if pre_end > 0:
                start = random.randint(0, pre_end)
            else:
                start = max(0, max_start - random.randint(0, min(max_start, int(0.1 * ci.sr))))

            start = int(np.clip(start, 0, max_start))
            x = y[start: start + self.base_W]
            if len(x) < self.base_W:
                x = np.pad(x, (0, self.base_W - len(x)))

            onset_offset_in_base = None
            is_pos = False

            if self.stage2:
                label = np.zeros(self.V, dtype=np.float32)
            else:
                # Stage1 should generally use p_neg=0; if you include negatives anyway, this is a placeholder
                label = 0

        # Augment base window -> crop (onset-aware) -> normalize what net sees
        x = x.astype(np.float32)
        x = self._augment(x)

        x, crop_start = self._maybe_crop(x, onset_offset_in_base=onset_offset_in_base, is_pos=is_pos)

        # Compute onset index inside the RETURNED window (after crop), for visualization
        onset_in_return = -1
        if is_pos and onset_offset_in_base is not None:
            onset_in_return = int(onset_offset_in_base - crop_start)
            if onset_in_return < 0 or onset_in_return >= len(x):
                onset_in_return = -1

        x = self._rms_norm(x)

        if self.preemph_coef > 0:
            x[1:] = x[1:] - self.preemph_coef * x[:-1]

        x_t = torch.from_numpy(x).unsqueeze(0)
        if self.stage2:
            y_t = torch.from_numpy(label)
        else:
            y_t = torch.tensor(label, dtype=torch.long)

        return x_t, y_t, int(onset_in_return)


# ----------------------------
# Model
# ----------------------------
class OnsetNet(nn.Module):
    """
    Tiny 1D conv. Accepts variable input length due to AdaptiveAvgPool1d(1).
    """
    def __init__(self, vocab_size: int, width: int = 64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, width // 2, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width // 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(width // 2, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(width, vocab_size)
        self.vocab_size = vocab_size
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x).squeeze(-1)
        return self.head(z)

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------
# Export to C header
# ----------------------------
def export_c_header(model: nn.Module, sr: int, vocab_size: int, export_window_samples: int, path: str, meta: Dict):
    model.eval()

    def ascii_pitch(p: str) -> str:
        return p.replace("\u266f", "#").replace("\u266d", "b")

    lines = [
        "// Auto-generated - DO NOT EDIT",
        "// OnsetNet weights for Daisy Seed bare metal inference",
        f"// sr={sr} vocab_size={vocab_size} window_samples={export_window_samples}",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define ONSET_SR           {sr}",
        f"#define ONSET_VOCAB_SIZE   {vocab_size}",
        f"#define ONSET_WINDOW       {export_window_samples}",
        f"#define ONSET_WIDTH        {model.width}",
        "",
        "static const char* onset_pitch_names[] = {",
    ]
    for p in meta.get("index_to_pitch", []):
        lines.append(f'    "{ascii_pitch(p)}",')
    lines += ["};", ""]

    def arr(name: str, t: torch.Tensor):
        flat = t.detach().cpu().float().numpy().flatten()
        vals = ", ".join(f"{v:.8f}f" for v in flat)
        return [
            f"// shape: {list(t.shape)}",
            f"static const float {name}[{len(flat)}] = {{",
            f"    {vals}",
            "};",
            "",
        ]

    sd = model.state_dict()
    layer_map = [
        ("conv.0",  "l0_conv"),
        ("conv.1",  "l0_bn"),
        ("conv.3",  "l1_conv"),
        ("conv.4",  "l1_bn"),
        ("conv.6",  "l2_conv"),
        ("conv.7",  "l2_bn"),
    ]

    for sd_prefix, c_prefix in layer_map:
        w_key = f"{sd_prefix}.weight"
        if w_key in sd:
            lines += arr(f"{c_prefix}_weight", sd[w_key])
        for suffix in ("bias", "running_mean", "running_var", "weight", "bias"):
            k = f"{sd_prefix}.{suffix}"
            if k in sd and not (suffix == "weight" and "conv" in sd_prefix and k == w_key):
                lines += arr(f"{c_prefix}_{suffix}", sd[k])

    lines += arr("head_weight", sd["head.weight"])
    lines += arr("head_bias",   sd["head.bias"])

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[INFO] Exported C header -> {path}")


# ----------------------------
# Resume helper
# ----------------------------
def try_resume(
    resume_path: str,
    model: nn.Module,
    opt: torch.optim.Optimizer,
    scaler: GradScaler,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    device: str,
):
    ckpt = torch.load(resume_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    if "optimizer_state" in ckpt:
        opt.load_state_dict(ckpt["optimizer_state"])
    if "scaler_state" in ckpt:
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            print("[WARN] Could not load scaler_state (ok if switching devices).")
    if scheduler is not None and "scheduler_state" in ckpt and ckpt["scheduler_state"] is not None:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            print("[WARN] Could not load scheduler_state.")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_score = float(ckpt.get("best_score", -1.0))
    stage2 = bool(ckpt.get("stage2", False))
    print(f"[INFO] Resumed from {resume_path} start_epoch={start_epoch} best_score={best_score:.3f} stage2={stage2}")
    return start_epoch, best_score, stage2


# ----------------------------
# Training helpers
# ----------------------------
def build_scheduler(args, opt):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * args.cosine_min_frac)
    if args.scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=args.step_size, gamma=args.step_gamma)
    raise ValueError(f"Unknown scheduler: {args.scheduler}")


def apply_curriculum(ds: OnsetDataset, ep: int, args):
    if args.curriculum_epochs <= 0:
        ds.set_aug_profile(
            noise_std=args.full_noise_std,
            gain_db_min=args.full_gain_min,
            gain_db_max=args.full_gain_max,
            enable_gain=args.full_enable_gain,
            enable_polarity=args.full_enable_polarity,
        )
        return

    if ep <= args.curriculum_epochs:
        ds.set_aug_profile(
            noise_std=args.calm_noise_std,
            gain_db_min=args.calm_gain_min,
            gain_db_max=args.calm_gain_max,
            enable_gain=args.calm_enable_gain,
            enable_polarity=args.calm_enable_polarity,
        )
    else:
        ds.set_aug_profile(
            noise_std=args.full_noise_std,
            gain_db_min=args.full_gain_min,
            gain_db_max=args.full_gain_max,
            enable_gain=args.full_enable_gain,
            enable_polarity=args.full_enable_polarity,
        )


# ----------------------------
# Training
# ----------------------------
def train(args):
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    clips, sr, vocab_size, meta = load_clips(args.dataset)

    # Windowing
    train_base_W = ms_to_samples(args.train_ms, sr)
    train_crop_W = ms_to_samples(args.crop_ms, sr)
    val_base_W   = ms_to_samples(args.val_ms, sr)
    val_crop_W   = ms_to_samples(args.val_crop_ms, sr)
    export_W     = ms_to_samples(args.export_ms, sr)

    print(
        f"[INFO] clips={len(clips)} sr={sr} vocab={vocab_size} "
        f"train_ms={args.train_ms:g} (base={train_base_W}) crop_ms={args.crop_ms:g} (crop={train_crop_W}) crop_prob={args.crop_prob:g} "
        f"val_ms={args.val_ms:g} (base={val_base_W}) val_crop_ms={args.val_crop_ms:g} (crop={val_crop_W}) val_crop_prob={args.val_crop_prob:g} "
        f"export_ms={args.export_ms:g} (exportW={export_W})"
    )

    stage2 = bool(args.stage2)

    if not args.include_chords:
        clips = [c for c in clips if c.clip_type != "chord"]
        print(f"[INFO] Stage filter (no chords): {len(clips)} clips")
    else:
        print(f"[INFO] include_chords enabled: {len(clips)} clips")

    train_clips, val_clips = train_val_split(clips, args.val_ratio, args.seed)
    print(f"[INFO] train={len(train_clips)} val={len(val_clips)}")

    model = OnsetNet(vocab_size=vocab_size, width=args.width).to(device)
    print(f"[INFO] params={model.count_params():,} width={args.width}")

    # Loss
    if stage2:
        pw = torch.full((vocab_size,), float(args.pos_weight), device=device)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
    else:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler(enabled=(device == "cuda"))
    scheduler = build_scheduler(args, opt)

    best_path = os.path.join(args.out, "onset_best.pt")
    last_path = os.path.join(args.out, "onset_last.pt")

    start_epoch = 1
    best_score = -1.0

    if args.resume:
        start_epoch, best_score, resumed_stage2 = try_resume(args.resume, model, opt, scaler, scheduler, device)
        if not args.force_stage and resumed_stage2 != stage2:
            stage2 = resumed_stage2
            print(f"[INFO] stage2 overridden by checkpoint: stage2={stage2}")
            if stage2:
                pw = torch.full((vocab_size,), float(args.pos_weight), device=device)
                loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)
            else:
                loss_fn = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # Datasets
    train_ds = OnsetDataset(
        train_clips,
        base_W=train_base_W,
        crop_W=train_crop_W,
        crop_prob=args.crop_prob,
        vocab_size=vocab_size,
        stage2=stage2,
        p_on=args.p_on, p_neg=args.p_neg,
        virtual_len=args.virtual_len,
        audio_cache_max=args.audio_cache_max,
        preemph_coef=args.preemph_coef,
        noise_std=args.calm_noise_std if args.curriculum_epochs > 0 else args.full_noise_std,
        gain_db_min=args.calm_gain_min if args.curriculum_epochs > 0 else args.full_gain_min,
        gain_db_max=args.calm_gain_max if args.curriculum_epochs > 0 else args.full_gain_max,
        enable_gain=args.calm_enable_gain if args.curriculum_epochs > 0 else args.full_enable_gain,
        enable_polarity=args.calm_enable_polarity if args.curriculum_epochs > 0 else args.full_enable_polarity,
        onset_pos_min=args.onset_pos_min,
        onset_pos_max=args.onset_pos_max,
        crop_keep_onset_prob=args.crop_keep_onset_prob,
    )

    val_ds = OnsetDataset(
        val_clips,
        base_W=val_base_W,
        crop_W=val_crop_W,
        crop_prob=args.val_crop_prob,
        vocab_size=vocab_size,
        stage2=stage2,
        p_on=args.val_p_on, p_neg=args.val_p_neg,
        virtual_len=max(10000, args.virtual_len // 10),
        audio_cache_max=64,
        preemph_coef=args.preemph_coef,
        noise_std=0.0,
        gain_db_min=0.0,
        gain_db_max=0.0,
        enable_gain=False,
        enable_polarity=False,
        onset_pos_min=args.onset_pos_min,
        onset_pos_max=args.onset_pos_max,
        crop_keep_onset_prob=args.crop_keep_onset_prob,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=pad_collate_1d_with_onset,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=max(1, args.workers // 2), pin_memory=True,
        persistent_workers=(args.workers > 0),
        collate_fn=pad_collate_1d_with_onset,
    )

    viz = None
    if args.viz:
        viz = LiveNetViz(
            model=model,
            meta=meta,
            sr=sr,
            thresh=args.thresh,
            topk=args.viz_topk,
            out_nodes_cap=args.viz_out_nodes,
            weight_viz_every=args.viz_weight_every,
            pulse_speed=args.viz_pulse_speed,
        )
        print("[INFO] LiveNetViz enabled.")

    def infinite(loader):
        while True:
            for b in loader:
                yield b

    train_it = infinite(train_loader)
    val_it   = infinite(val_loader)

    th_grid = [round(x, 2) for x in np.linspace(args.th_sweep_min, args.th_sweep_max, args.th_sweep_steps)]
    global_step = 0

    try:
        for ep in range(start_epoch, args.epochs + 1):
            apply_curriculum(train_ds, ep, args)

            model.train()
            t0 = time.time()
            tr_loss = 0.0

            viz_every = max(1, int(args.viz_every)) if viz is not None else 0

            # ------------------------
            # Train
            # ------------------------
            for step in range(args.steps_per_epoch):
                x, y, onset_idx = next(train_it)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if args.warmup_steps > 0:
                    maybe_linear_warmup_lr(opt, args.lr, global_step, args.warmup_steps)

                opt.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", enabled=(device == "cuda")):
                    logits = model(x)
                    if stage2:
                        loss = loss_fn(logits, y)
                    else:
                        loss = loss_fn(logits, y.view(-1))

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                scaler.step(opt)
                scaler.update()

                tr_loss += float(loss.item())
                global_step += 1

                if viz is not None and (step % viz_every == 0):
                    viz.update(x_batch=x, y_info=y, onset_idx=onset_idx, device=device, stage2=stage2)

            tr_loss /= args.steps_per_epoch
            if scheduler is not None and args.scheduler != "none":
                scheduler.step()

            # ------------------------
            # Validation
            # ------------------------
            model.eval()
            v_loss = 0.0
            all_logits, all_targets = [], []

            with torch.no_grad():
                for _ in range(args.val_steps):
                    x, y, onset_idx = next(val_it)
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    logits = model(x)
                    if stage2:
                        v_loss += float(loss_fn(logits, y).item())
                        all_logits.append(logits.cpu())
                        all_targets.append(y.cpu())
                    else:
                        v_loss += float(loss_fn(logits, y.view(-1)).item())
                        all_logits.append(logits.cpu())
                        all_targets.append(y.view(-1).cpu())

            v_loss /= args.val_steps
            all_logits = torch.cat(all_logits)
            all_targets = torch.cat(all_targets)

            lr_now = opt.param_groups[0]["lr"]
            dt = time.time() - t0

            if stage2:
                best_f1, best_th, best_p, best_r = best_f1_over_thresholds(all_logits, all_targets, th_grid)
                p_fix, r_fix, f1_fix = f1_scores_from_logits(all_logits, all_targets, thresh=args.thresh)
                metric_str = (
                    f"bestF1={best_f1:.3f}@{best_th:.2f} (P={best_p:.3f} R={best_r:.3f}) "
                    f"fixF1={f1_fix:.3f}@{args.thresh:.2f}"
                )
                score_for_best = best_f1
            else:
                probs = torch.softmax(all_logits, dim=1)
                top1 = (probs.argmax(dim=1) == all_targets).float().mean().item()
                top3 = (probs.topk(k=min(3, probs.shape[1]), dim=1).indices == all_targets.unsqueeze(1)).any(dim=1).float().mean().item()
                mean_true_p = probs[torch.arange(probs.shape[0]), all_targets].mean().item()
                metric_str = f"top1={top1:.3f} top3={top3:.3f} mean_true_p={mean_true_p:.3f}"
                score_for_best = top1

            aug_profile = (
                f"aug(noise={train_ds.noise_std:g}, gain=[{train_ds.gain_db_min:g},{train_ds.gain_db_max:g}], "
                f"pol={'on' if train_ds.enable_polarity else 'off'})"
            )
            win_profile = (
                f"win(train {args.train_ms:g}ms->crop {args.crop_ms:g}ms p={args.crop_prob:g}; "
                f"val {args.val_ms:g}ms->crop {args.val_crop_ms:g}ms p={args.val_crop_prob:g}) "
                f"anchor([{args.onset_pos_min:g},{args.onset_pos_max:g}] keep_crop_onset={args.crop_keep_onset_prob:g})"
            )

            print(
                f"[EP {ep:03d}] train={tr_loss:.4f} val={v_loss:.4f} {metric_str} "
                f"lr={lr_now:.2e} {aug_profile} {win_profile} t={dt:.1f}s"
            )

            # Save checkpoints
            ckpt = {
                "epoch": ep,
                "stage2": stage2,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "scheduler_state": (scheduler.state_dict() if scheduler is not None else None),
                "best_score": best_score,
                "sr": sr,
                "vocab_size": vocab_size,
                "export_window_samples": export_W,
                "width": args.width,
                "thresh": args.thresh,
                "args": vars(args),
            }
            torch.save(ckpt, last_path)

            if score_for_best > best_score:
                best_score = score_for_best
                ckpt["best_score"] = best_score
                torch.save(ckpt, best_path)
                print(f"[INFO] New best score={best_score:.3f} -> {best_path}")

                c_header_path = os.path.join(args.out, "onset_weights.h")
                export_c_header(model, sr, vocab_size, export_W, c_header_path, meta)

        print(f"\nDone. Best score={best_score:.3f}")
        print(f"Weights: {best_path}")
        print(f"C header: {os.path.join(args.out, 'onset_weights.h')}")

    finally:
        if viz is not None:
            viz.close()


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset", type=str, default="dataset")
    ap.add_argument("--out", type=str, default="checkpoints")
    ap.add_argument("--resume", type=str, default="", help="Path to .pt checkpoint to resume from")

    # Stage controls
    ap.add_argument("--stage2", action="store_true", help="Stage2: multi-label BCE")
    ap.add_argument("--include_chords", action="store_true", help="Include chord clips")
    ap.add_argument("--force_stage", action="store_true", help="Do not override stage from checkpoint when resuming")

    # Training loop
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--steps_per_epoch", type=int, default=400)
    ap.add_argument("--val_steps", type=int, default=120)
    ap.add_argument("--val_ratio", type=float, default=0.1)

    # Optimization
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--clip_grad", type=float, default=1.0)
    ap.add_argument("--warmup_steps", type=int, default=0)
    ap.add_argument("--label_smoothing", type=float, default=0.0)

    # Scheduler
    ap.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine", "none"])
    ap.add_argument("--step_size", type=int, default=25)
    ap.add_argument("--step_gamma", type=float, default=0.5)
    ap.add_argument("--cosine_min_frac", type=float, default=0.01)

    # Model
    ap.add_argument("--width", type=int, default=64)

    # Data loading
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--audio_cache_max", type=int, default=256)
    ap.add_argument("--virtual_len", type=int, default=100000)

    # Sampling ratios
    ap.add_argument("--p_on", type=float, default=1.0)
    ap.add_argument("--p_neg", type=float, default=0.0)
    ap.add_argument("--val_p_on", type=float, default=1.0)
    ap.add_argument("--val_p_neg", type=float, default=0.0)

    # Preprocess
    ap.add_argument("--preemph_coef", type=float, default=0.97)

    # Stage2 BCE
    ap.add_argument("--pos_weight", type=float, default=2.0)

    # Curriculum aug
    ap.add_argument("--curriculum_epochs", type=int, default=15)
    ap.add_argument("--calm_noise_std", type=float, default=0.0)
    ap.add_argument("--calm_gain_min", type=float, default=-4.0)
    ap.add_argument("--calm_gain_max", type=float, default=2.0)
    ap.add_argument("--calm_enable_gain", type=int, default=1)
    ap.add_argument("--calm_enable_polarity", type=int, default=0)

    ap.add_argument("--full_noise_std", type=float, default=0.0005)
    ap.add_argument("--full_gain_min", type=float, default=-18.0)
    ap.add_argument("--full_gain_max", type=float, default=6.0)
    ap.add_argument("--full_enable_gain", type=int, default=1)
    ap.add_argument("--full_enable_polarity", type=int, default=1)

    # Context + crop (defaults preserve old 10ms->10ms behavior)
    ap.add_argument("--train_ms", type=float, default=DEFAULT_INFER_MS)
    ap.add_argument("--crop_ms", type=float, default=DEFAULT_INFER_MS)
    ap.add_argument("--crop_prob", type=float, default=1.0)

    ap.add_argument("--val_ms", type=float, default=DEFAULT_INFER_MS)
    ap.add_argument("--val_crop_ms", type=float, default=DEFAULT_INFER_MS)
    ap.add_argument("--val_crop_prob", type=float, default=1.0)

    ap.add_argument("--export_ms", type=float, default=DEFAULT_INFER_MS)

    # Onset anchoring (positives)
    ap.add_argument("--onset_pos_min", type=float, default=0.20)
    ap.add_argument("--onset_pos_max", type=float, default=0.65)
    ap.add_argument("--crop_keep_onset_prob", type=float, default=0.90)

    # Thresholds (viz + stage2)
    ap.add_argument("--thresh", type=float, default=0.2)
    ap.add_argument("--th_sweep_min", type=float, default=0.05)
    ap.add_argument("--th_sweep_max", type=float, default=0.95)
    ap.add_argument("--th_sweep_steps", type=int, default=19)

    # Live viz
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--viz_every", type=int, default=10)
    ap.add_argument("--viz_topk", type=int, default=8)
    ap.add_argument("--viz_out_nodes", type=int, default=56)
    ap.add_argument("--viz_weight_every", type=int, default=10)
    ap.add_argument("--viz_pulse_speed", type=float, default=0.5)

    args = ap.parse_args()

    # Convert int flags to bools
    args.calm_enable_gain = bool(args.calm_enable_gain)
    args.calm_enable_polarity = bool(args.calm_enable_polarity)
    args.full_enable_gain = bool(args.full_enable_gain)
    args.full_enable_polarity = bool(args.full_enable_polarity)

    train(args)


if __name__ == "__main__":
    main()