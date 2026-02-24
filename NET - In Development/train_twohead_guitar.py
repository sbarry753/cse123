"""
Onset-only training for <10ms guitar note detection on Daisy Seed.

Architecture:
  - OnsetNet: tiny conv, 480 samples (10ms @ 48kHz) input
  - Single head: 55-note multi-hot (but stage1 is single notes only)

Key design decisions:
  - Train ONLY on transient windows (first 10ms after pick attack)
  - Heavy gain augmentation to handle real-world dynamics
  - pos_weight=54.0 to counteract 1/55 class imbalance
  - No hold head (separate concern for sustain tracking)

Export:
  - Weights exported as raw C float arrays for bare metal Daisy inference

Extras (this version):
  - LiveNetViz: wiring-diagram style live visualizer while training
    * black bg, white node rings
    * green/red edges by weight sign
    * node glow by activation magnitude
    * waveform + top-k predictions
    * TRUE current input note (from y)
    * ANIMATION: "signal path" pulse from input -> conv0 -> conv1 -> conv2 -> output
        - chooses the most “active” channel in each layer (by activation magnitude)
        - chooses the most likely output (top predicted note)
        - draws a bright pulsing path through those nodes

Resume:
  - --resume path/to/onset_last.pt
  - Restores model/optimizer/scaler/scheduler and continues epochs

Command line usage:
    python train_twohead_guitar.py --dataset labels --out checkpoints --viz --viz_every 10 --viz_weight_every 3 --viz_pulse_speed 0.45 --resume checkpoints/onset_best.pt
"""

import os
import json
import time
import random
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import soundfile as sf

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler


# ----------------------------
# Constants
# ----------------------------
WINDOW_MS = 10.0          # inference window on Daisy
PRE_ROLL_MS = 2.0         # how much before onset to include


# ----------------------------
# Utilities
# ----------------------------
def set_seed(seed: int):
    """Make training reproducible-ish (still nondeterministic on GPU sometimes)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> Dict[str, Any]:
    """Load a JSON file into a python dict."""
    with open(path, "r") as f:
        return json.load(f)


def read_manifest(path: str) -> List[Dict[str, str]]:
    """Read manifest.jsonl (one JSON object per line)."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def f1_scores(logits: torch.Tensor, targets: torch.Tensor, thresh: float = 0.5):
    """
    Micro-averaged precision/recall/F1 over the whole batch (multi-label).
    Uses sigmoid(logits) and compares to `thresh`.
    """
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


def label_to_note_name(label_vec: torch.Tensor, meta: Dict[str, Any]) -> str:
    """
    label_vec: (V,) multi-hot float tensor
    Returns "E2" or "E2,G2" or "(none)".
    Uses meta["index_to_pitch"] if present.
    """
    pitches = meta.get("index_to_pitch", None)
    idxs = (label_vec >= 0.5).nonzero(as_tuple=False).flatten().tolist()
    if not idxs:
        return "(none)"  # negative/no-onset sample
    if pitches is None:
        return ",".join(str(i) for i in idxs)
    names = [pitches[i] if i < len(pitches) else str(i) for i in idxs]
    return ",".join(names)


def top_pred_note_name(logits_1: torch.Tensor, meta: Dict[str, Any]) -> Tuple[int, float, str]:
    """
    logits_1: (V,) logits for one sample
    Returns: (idx, prob, name)
    """
    probs = torch.sigmoid(logits_1).detach().cpu().float().numpy()
    j = int(np.argmax(probs))
    p = float(probs[j])
    pitches = meta.get("index_to_pitch", None)
    name = pitches[j] if (pitches is not None and j < len(pitches)) else str(j)
    return j, p, name


# ----------------------------
# Live Net Visualizer (Matplotlib) + Animated Path
# ----------------------------
# Wiring diagram style:
#   - black background
#   - white node rings
#   - green/red edges by weight sign
#   - edge thickness/alpha by |weight|
#   - node glow by activation magnitude (forward hooks)
#   - side panels: waveform + top-k predictions + TRUE note label
#   - animated "signal path" pulse from input->...->output
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D


class LiveNetViz:
    """
    Wiring-diagram style graph visualizer for OnsetNet:
        Conv1D -> Conv1D -> Conv1D -> Linear

    Animation:
      - We pick a "path" each update:
          input_bucket (by waveform energy) ->
          most active c0 channel ->
          most active c1 channel ->
          most active c2 channel ->
          top predicted output class
      - Then we draw a bright pulsing path along those connections.
    """

    def __init__(
        self,
        model: nn.Module,
        meta: Dict[str, Any],
        sr: int,
        window_samples: int,
        thresh: float = 0.5,
        topk: int = 8,
        out_nodes_cap: int = 55,
        weight_viz_every: int = 1,   # update weight-colored edges every N viz updates (perf knob)
        pulse_speed: float = 0.35,   # pulse position step per update
    ):
        self.model = model
        self.meta = meta or {}
        self.sr = int(sr)
        self.W = int(window_samples)
        self.thresh = float(thresh)
        self.topk = int(topk)
        self.pitches = self.meta.get("index_to_pitch", None)

        self.weight_viz_every = max(1, int(weight_viz_every))
        self.pulse_speed = float(pulse_speed)

        # Expect OnsetNet layout:
        # conv = [Conv0, BN0, ReLU, Conv1, BN1, ReLU, Conv2, BN2, ReLU, AdaptiveAvgPool]
        self.conv0 = model.conv[0]
        self.conv1 = model.conv[3]
        self.conv2 = model.conv[6]
        self.pool  = model.conv[9]
        self.head  = model.head

        # Node counts (draw each conv output channel as a node)
        self.n_in = 8  # stylized input column (bucketed energy)
        self.n_c0 = int(self.conv0.out_channels)
        self.n_c1 = int(self.conv1.out_channels)
        self.n_c2 = int(self.conv2.out_channels)
        self.n_out_full = int(self.head.out_features)
        self.n_out = min(out_nodes_cap, self.n_out_full)

        # Activations from hooks
        self.act_c0 = None
        self.act_c1 = None
        self.act_c2 = None
        self.act_h  = None
        self._install_hooks()

        # Animation state
        self._viz_tick = 0
        self._pulse_pos = 0.0  # 0..4 (segments: in->c0->c1->c2->out)
        self._last_path_nodes: Optional[Dict[str, int]] = None

        # Matplotlib figure
        plt.ion()
        self.fig = plt.figure(figsize=(14, 6), facecolor="black")
        gs = self.fig.add_gridspec(2, 2, width_ratios=[2.2, 1.0], height_ratios=[1, 1])

        self.ax_net  = self.fig.add_subplot(gs[:, 0])
        self.ax_wave = self.fig.add_subplot(gs[0, 1])
        self.ax_pred = self.fig.add_subplot(gs[1, 1])

        for ax in (self.ax_net, self.ax_wave, self.ax_pred):
            ax.set_facecolor("black")
        self.ax_net.set_axis_off()

        # Layout + artists
        self.pos = self._make_positions()
        self.node_artists: Dict[Tuple[str, int], Circle] = {}
        self.edge_artists: List[Tuple[str, str, Tuple[str, int], Tuple[str, int], Line2D]] = []

        # Fast lookup: (a, b, src_idx, dst_idx) -> Line2D
        self.edge_lookup: Dict[Tuple[str, str, int, int], Line2D] = {}

        self._build_static_artists()

        # Cache “base” edge styling so the pulse can override temporarily
        self._edge_base_style: Dict[Line2D, Tuple[Tuple[float, float, float], float, float]] = {}

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
        def hook_pool(_, __, out): self.act_h = out.detach()

        self.conv0.register_forward_hook(hook_c0)
        self.conv1.register_forward_hook(hook_c1)
        self.conv2.register_forward_hook(hook_c2)
        self.pool.register_forward_hook(hook_pool)

    def _make_positions(self):
        """Place nodes in 5 columns: input -> c0 -> c1 -> c2 -> out."""
        pos: Dict[Tuple[str, int], Tuple[float, float]] = {}
        cols = [
            ("in",  self.n_in),
            ("c0",  self.n_c0),
            ("c1",  self.n_c1),
            ("c2",  self.n_c2),
            ("out", self.n_out),
        ]
        xs = [0.05, 0.30, 0.52, 0.74, 0.95]
        for (name, n), x in zip(cols, xs):
            if n <= 1:
                ys = [0.5]
            else:
                ys = list(reversed([0.1 + 0.8 * (i/(n-1)) for i in range(n)]))
            for i, y in enumerate(ys):
                pos[(name, i)] = (x, y)
        return pos

    def _build_static_artists(self):
        """Create node circles + dense edges between adjacent columns."""
        def add_node(key, radius=0.018):
            x, y = self.pos[key]
            circ = Circle(
                (x, y),
                radius=radius,
                facecolor=(0, 0, 0, 1),
                edgecolor=(1, 1, 1, 1),
                linewidth=2.0,
            )
            self.ax_net.add_patch(circ)
            self.node_artists[key] = circ

        # Nodes
        for i in range(self.n_in): add_node(("in", i), radius=0.020)
        for i in range(self.n_c0): add_node(("c0", i), radius=0.018)
        for i in range(self.n_c1): add_node(("c1", i), radius=0.018)
        for i in range(self.n_c2): add_node(("c2", i), radius=0.018)
        for i in range(self.n_out): add_node(("out", i), radius=0.016)

        # Edges
        self.edge_artists = []
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
        """
        Aggregate weights to match our drawn layer sizes:
          - in->c0: conv0.weight (C0,1,K) -> mean over kernel -> (C0,)
          - c0->c1: conv1.weight (C1,C0,K) -> mean over kernel -> (C1,C0)
          - c1->c2: conv2.weight (C2,C1,K) -> mean over kernel -> (C2,C1)
          - c2->out: head.weight (V,C2) -> slice top n_out -> (n_out,C2)
        """
        w0 = self.conv0.weight.detach().float().mean(dim=2).squeeze(1)  # (C0,)
        w1 = self.conv1.weight.detach().float().mean(dim=2)             # (C1,C0)
        w2 = self.conv2.weight.detach().float().mean(dim=2)             # (C2,C1)
        wh = self.head.weight.detach().float()[: self.n_out]            # (n_out,C2)
        return w0, w1, w2, wh

    def _get_act_vecs(self):
        """Per-channel activation magnitudes for glow: (B,C,T)->mean(|.|) over (B,T)->(C,)"""
        def chan_mag(t):
            if t is None:
                return None
            return t.detach().float().abs().mean(dim=(0, 2)).cpu()
        a0 = chan_mag(self.act_c0)
        a1 = chan_mag(self.act_c1)
        a2 = chan_mag(self.act_c2)
        return a0, a1, a2

    def _update_nodes_from_activations(self):
        """Base node glow from activation magnitudes."""
        a0, a1, a2 = self._get_act_vecs()

        def set_glow(layer, mags):
            if mags is None:
                return
            mags = mags / (mags.max().clamp(min=1e-6))
            for i in range(len(mags)):
                key = (layer, i)
                if key in self.node_artists:
                    g = float(mags[i].item())
                    # subtle green glow
                    self.node_artists[key].set_facecolor((0.10, 0.60, 0.10, 0.06 + 0.55 * g))

        set_glow("c0", a0)
        set_glow("c1", a1)
        set_glow("c2", a2)

    def _update_edges_from_weights(self):
        """Base edge styling from weights (green/red + thickness)."""
        w0, w1, w2, wh = self._get_weight_mats()

        def norm_abs(x):
            x = x.abs()
            return x / (x.max().clamp(min=1e-6))

        w0n = norm_abs(w0.cpu())
        w1n = norm_abs(w1.cpu())
        w2n = norm_abs(w2.cpu())
        whn = norm_abs(wh.cpu())

        for a, b, ka, kb, line in self.edge_artists:
            ia = ka[1]
            ib = kb[1]

            if a == "in" and b == "c0":
                w = float(w0[ib].cpu().item())
                s = float(w0n[ib].item())
            elif a == "c0" and b == "c1":
                w = float(w1[ib, ia].cpu().item())
                s = float(w1n[ib, ia].item())
            elif a == "c1" and b == "c2":
                w = float(w2[ib, ia].cpu().item())
                s = float(w2n[ib, ia].item())
            elif a == "c2" and b == "out":
                w = float(wh[ib, ia].cpu().item())
                s = float(whn[ib, ia].item())
            else:
                continue

            r, g, bl = self._color_from_weight(w)
            lw = 0.12 + 2.2 * s
            al = 0.02 + 0.55 * s

            line.set_color((r, g, bl))
            line.set_linewidth(lw)
            line.set_alpha(al)

            # cache base style so pulses can temporarily override and then revert
            self._edge_base_style[line] = ((r, g, bl), lw, al)

    def _choose_input_bucket(self, x_np: np.ndarray) -> int:
        """
        Pick one of self.n_in "input nodes" based on energy in that segment.
        This is purely visual (since input isn't actually 8 nodes).
        """
        if x_np.size == 0:
            return 0
        segs = np.array_split(x_np, self.n_in)
        energies = np.array([float(np.mean(s * s)) if s.size else 0.0 for s in segs], dtype=np.float32)
        return int(np.argmax(energies))

    def _choose_path(self, x_1: torch.Tensor, logits_1: torch.Tensor) -> Dict[str, int]:
        """
        Decide which nodes are "active" and define a path:
          in -> c0 -> c1 -> c2 -> out
        """
        x_np = x_1.detach().float().cpu().numpy().reshape(-1)
        in_idx = self._choose_input_bucket(x_np)

        a0, a1, a2 = self._get_act_vecs()
        c0_idx = int(torch.argmax(a0).item()) if a0 is not None else 0
        c1_idx = int(torch.argmax(a1).item()) if a1 is not None else 0
        c2_idx = int(torch.argmax(a2).item()) if a2 is not None else 0

        out_idx_full = int(torch.argmax(torch.sigmoid(logits_1)).item())
        out_idx = min(out_idx_full, self.n_out - 1)

        return {"in": in_idx, "c0": c0_idx, "c1": c1_idx, "c2": c2_idx, "out": out_idx}

    def _revert_previous_pulse(self):
        """Revert previously pulsed edges back to their base style."""
        if self._last_path_nodes is None:
            return
        path = self._last_path_nodes
        segs = [
            ("in", "c0", path["in"],  path["c0"]),
            ("c0", "c1", path["c0"],  path["c1"]),
            ("c1", "c2", path["c1"],  path["c2"]),
            ("c2", "out", path["c2"], path["out"]),
        ]
        for a, b, si, di in segs:
            line = self.edge_lookup.get((a, b, si, di))
            if line is None:
                continue
            base = self._edge_base_style.get(line)
            if base is None:
                continue
            col, lw, al = base
            line.set_color(col)
            line.set_linewidth(lw)
            line.set_alpha(al)

        # revert node facecolors on the path nodes back to black fill (but keep activation glows)
        # we won't hard-reset node colors here because activation glow is the base; we only boost later.

    def _apply_pulse(self, path: Dict[str, int]):
        """
        Apply a bright pulsing highlight along the chosen path.
        Pulse position moves from 0->4 across the segments.
        """
        # Revert last pulse first (so we don’t accumulate overrides)
        self._revert_previous_pulse()

        # Update pulse pos
        self._pulse_pos += self.pulse_speed
        if self._pulse_pos >= 4.0:
            self._pulse_pos = 0.0

        # Path segments (index 0..3)
        segs = [
            ("in", "c0", path["in"],  path["c0"]),
            ("c0", "c1", path["c0"],  path["c1"]),
            ("c1", "c2", path["c1"],  path["c2"]),
            ("c2", "out", path["c2"], path["out"]),
        ]

        # Determine which segment is currently “hot”
        hot_seg = int(np.floor(self._pulse_pos))
        frac = float(self._pulse_pos - hot_seg)  # 0..1 within the segment

        # Pulse intensity: ramp up/down smoothly
        pulse = 0.5 + 0.5 * np.sin(2.0 * np.pi * frac)  # 0..1

        for si, (a, b, src_i, dst_i) in enumerate(segs):
            line = self.edge_lookup.get((a, b, src_i, dst_i))
            if line is None:
                continue

            base = self._edge_base_style.get(line)
            if base is None:
                # fallback base style
                base_col, base_lw, base_al = ((0.35, 0.35, 0.35), 0.5, 0.08)
            else:
                base_col, base_lw, base_al = base

            # If segment is before the hot segment, keep it “lit”
            if si < hot_seg:
                boost = 0.85
            # Hot segment pulses
            elif si == hot_seg:
                boost = 0.35 + 0.65 * pulse
            else:
                boost = 0.0

            # Apply bright cyan-ish pulse overlay (like a signal)
            # (We keep weight sign color as base, but punch alpha/width up.)
            # If you want strictly cyan, replace base_col with (0.4, 0.9, 1.0).
            col = base_col
            lw = base_lw + 5.0 * boost
            al = min(1.0, base_al + 0.85 * boost)

            line.set_color(col)
            line.set_linewidth(lw)
            line.set_alpha(al)

        # Also boost the path nodes (white-ish glow)
        for layer in ("in", "c0", "c1", "c2", "out"):
            idx = path[layer]
            key = (layer, idx)
            if key in self.node_artists:
                # brighten fill slightly to show “signal”
                # input nodes: bluish, hidden layers: greenish, output: yellowish
                if layer == "in":
                    self.node_artists[key].set_facecolor((0.10, 0.55, 0.70, 0.85))
                elif layer == "out":
                    self.node_artists[key].set_facecolor((0.80, 0.75, 0.15, 0.85))
                else:
                    self.node_artists[key].set_facecolor((0.15, 0.75, 0.20, 0.85))

        self._last_path_nodes = dict(path)

    def _update_wave_and_preds(self, x_1: torch.Tensor, logits_1: torch.Tensor, y_1: torch.Tensor):
        """Update right-side panels: waveform + top-k prediction bar chart."""
        x_np = x_1.detach().float().cpu().numpy().reshape(-1)
        probs = torch.sigmoid(logits_1.detach()).cpu().numpy().reshape(-1)

        true_note = label_to_note_name(y_1.detach().cpu(), self.meta)
        out_j, out_p, out_name = top_pred_note_name(logits_1.detach().cpu(), self.meta)

        # Waveform
        self.ax_wave.clear()
        self.ax_wave.set_facecolor("black")
        self.ax_wave.set_title(f"Input (10ms) — TRUE: {true_note} | PRED: {out_name} ({out_p:.2f})", color="white")
        t = np.arange(len(x_np)) / float(self.sr) * 1000.0
        self.ax_wave.plot(t, x_np, linewidth=1.0)
        self.ax_wave.set_xlim(0, t[-1] if len(t) else 1.0)
        self.ax_wave.tick_params(colors="white")
        for spine in self.ax_wave.spines.values():
            spine.set_color("white")

        # Predictions
        self.ax_pred.clear()
        self.ax_pred.set_facecolor("black")
        self.ax_pred.set_title("Top predictions", color="white")

        topk = min(self.topk, len(probs))
        idxs = np.argsort(-probs)[:topk]
        vals = probs[idxs]

        labels = []
        for j in idxs:
            if self.pitches is not None and j < len(self.pitches):
                labels.append(self.pitches[j])
            else:
                labels.append(str(j))

        self.ax_pred.barh(range(topk)[::-1], vals[::-1])
        self.ax_pred.set_yticks(range(topk)[::-1])
        self.ax_pred.set_yticklabels(labels[::-1], color="white")
        self.ax_pred.set_xlim(0.0, 1.0)
        self.ax_pred.axvline(self.thresh, linestyle="--", linewidth=1.0)
        self.ax_pred.tick_params(colors="white")
        for spine in self.ax_pred.spines.values():
            spine.set_color("white")

    @torch.no_grad()
    def update(self, x_batch: torch.Tensor, y_batch: torch.Tensor, device: str):
        """
        Call during training.
        Runs a forward pass on ONE sample and updates:
          - base weight edges (occasionally, for perf)
          - node activation glow
          - animated pulse path
          - waveform + prediction panels
        """
        self.model.eval()

        # Take a single sample for viz (keeps it fast)
        x = x_batch[:1].to(device, non_blocking=True)
        y = y_batch[:1].detach().cpu()

        # Forward pass triggers hooks (activations)
        logits = self.model(x)  # (1, V)

        self._viz_tick += 1

        # Update base weight edges occasionally (this is the expensive part)
        if (self._viz_tick % self.weight_viz_every) == 0 or not self._edge_base_style:
            self._update_edges_from_weights()

        # Base node glows from activations
        self._update_nodes_from_activations()

        # Choose an animated path and apply pulse highlight
        path = self._choose_path(x_1=x[0], logits_1=logits[0])
        self._apply_pulse(path)

        # Side panels
        self._update_wave_and_preds(x_1=x[0], logits_1=logits[0], y_1=y[0])

        # Render
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
    onset_sample: int       # sample index of pick attack in the clip
    transient_end: int      # end of labeled transient region
    num_samples: int
    sr: int
    clip_type: str          # "single", "scale_segment", "chord"


def load_clips(dataset_root: str) -> Tuple[List[ClipInfo], int, int]:
    """Load metadata.json + manifest.jsonl + per-item labels."""
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
        clip_type = lab.get("type", "unknown")

        clips.append(ClipInfo(
            audio_abs=audio_abs,
            multi_hot=mh,
            onset_sample=onset,
            transient_end=t_end,
            num_samples=n,
            sr=sr,
            clip_type=clip_type,
        ))

    return clips, sr, vocab_size


def train_val_split(clips: List[ClipInfo], val_ratio: float, seed: int):
    """Random split clips into train/val sets."""
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
    Every sample is a W-sample window centered on a pick transient.

    Modes:
      "on"  (p_on):  window overlaps transient -> label = multi_hot
      "neg" (p_neg): window from silence before transient OR after sustain -> label = zeros

    Augmentations applied every sample:
      - Random gain (-18 to +6 dB)
      - Additive white noise (very low level)
      - Random polarity flip
      - RMS normalization
      - Optional pre-emphasis
    """

    def __init__(
        self,
        clips: List[ClipInfo],
        window_samples: int,
        p_on: float,
        p_neg: float,
        virtual_len: int = 100000,
        audio_cache_max: int = 256,
        preemph_coef: float = 0.0,    # 0 = disabled
        noise_std: float = 0.001,
    ):
        self.clips = clips
        self.W = int(window_samples)
        self.virtual_len = int(virtual_len)
        self.preemph_coef = float(preemph_coef)
        self.noise_std = float(noise_std)

        # Normalize p_on/p_neg so they sum to 1.0
        s = p_on + p_neg
        self.p_on  = p_on / s
        self.p_neg = p_neg / s

        # Only clips with a clear transient window available
        self.on_candidates = [
            i for i, c in enumerate(clips)
            if c.transient_end > c.onset_sample
        ]

        # Simple audio cache to avoid re-reading the same files constantly
        self.audio_cache: Dict[str, np.ndarray] = {}
        self.cache_order: List[str] = []
        self.cache_max = int(audio_cache_max)

    def __len__(self):
        # Virtual length means we can sample "infinite" jittered windows
        return self.virtual_len

    def _load(self, path: str, sr_expected: int) -> np.ndarray:
        """Load mono float32 audio (cached)."""
        if path in self.audio_cache:
            return self.audio_cache[path]

        y, sr = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = np.mean(y, axis=1)

        assert int(sr) == int(sr_expected), f"SR mismatch {path}"

        self.audio_cache[path] = y
        self.cache_order.append(path)

        # Evict oldest cache item if we exceed max size
        if len(self.cache_order) > self.cache_max:
            old = self.cache_order.pop(0)
            self.audio_cache.pop(old, None)

        return y

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Gain, noise, polarity flip."""
        # Gain (-18 to +6 dB)
        gain_db = random.uniform(-18.0, 6.0)
        x = x * (10.0 ** (gain_db / 20.0))

        # Clip to [-1, 1] (simulate preamp saturation)
        x = np.clip(x, -1.0, 1.0)

        # White noise
        if self.noise_std > 0:
            x = x + np.random.randn(len(x)).astype(np.float32) * self.noise_std

        # Polarity flip
        if random.random() < 0.5:
            x = -x

        return x

    def _rms_norm(self, x: np.ndarray) -> np.ndarray:
        """Normalize each window to constant RMS (helps dynamics invariance)."""
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        return x / max(rms, 1e-4)

    def __getitem__(self, _):
        """
        Sample either:
          - a positive transient window (label = multi_hot)
          - a negative window from silence (label = zeros)
        """
        mode = "on" if random.random() < self.p_on else "neg"

        if mode == "on" and self.on_candidates:
            ci = self.clips[random.choice(self.on_candidates)]
            y  = self._load(ci.audio_abs, ci.sr)

            # Window must overlap the transient.
            # Place the onset somewhere inside the window with jitter.
            max_start = max(0, ci.num_samples - self.W)

            # Onset lands at a random position in the window
            pre = int(random.uniform(0, self.W * 0.7))
            start = max(0, min(ci.onset_sample - pre, max_start))

            x = y[start: start + self.W]
            if len(x) < self.W:
                x = np.pad(x, (0, self.W - len(x)))

            label = np.array(ci.multi_hot, dtype=np.float32)

        else:
            # Negative: silence before transient OR after sustain decays
            ci = self.clips[random.randrange(len(self.clips))]
            y  = self._load(ci.audio_abs, ci.sr)

            max_start = max(0, ci.num_samples - self.W)

            # Try pre-transient silence region first
            pre_end = max(0, ci.onset_sample - self.W)
            if pre_end > 0:
                start = random.randint(0, pre_end)
            else:
                # Pull from end of file (post-sustain)
                start = max(0, max_start - random.randint(0, min(max_start, int(0.1 * ci.sr))))

            start = int(np.clip(start, 0, max_start))
            x = y[start: start + self.W]
            if len(x) < self.W:
                x = np.pad(x, (0, self.W - len(x)))

            label = np.zeros(len(ci.multi_hot), dtype=np.float32)

        # Float + augmentation + normalization
        x = x.astype(np.float32)
        x = self._augment(x)
        x = self._rms_norm(x)

        # Optional pre-emphasis
        if self.preemph_coef > 0:
            x[1:] = x[1:] - self.preemph_coef * x[:-1]

        return (
            torch.from_numpy(x).unsqueeze(0),  # (1, W)
            torch.from_numpy(label),           # (vocab_size,)
        )


# ----------------------------
# Model
# ----------------------------
class OnsetNet(nn.Module):
    """
    Tiny 1D conv sized for 480 samples (10ms @ 48kHz).

    MAC count estimate:
      Conv(1,16,9,s2):  9*1*16*240   =   34,560
      Conv(16,32,5,s2): 5*16*32*120  =  307,200
      Conv(32,32,5,s2): 5*32*32*60   =  307,200
      Linear(32,55):    32*55         =    1,760
      Total: ~650K MACs — fits in Daisy 10ms budget
    """

    def __init__(self, vocab_size: int, width: int = 32):
        super().__init__()
        self.conv = nn.Sequential(
            # (1, 480) -> (16, 240)
            nn.Conv1d(1, width // 2, kernel_size=9, stride=2, padding=4, bias=False),
            nn.BatchNorm1d(width // 2),
            nn.ReLU(inplace=True),

            # (16, 240) -> (32, 120)
            nn.Conv1d(width // 2, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            # (32, 120) -> (32, 60)
            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            # (32, 60) -> (32, 1)
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(width, vocab_size)
        self.vocab_size = vocab_size
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, W) -> logits: (B, vocab_size)"""
        z = self.conv(x).squeeze(-1)  # (B, width)
        return self.head(z)           # (B, vocab_size)

    def count_params(self) -> int:
        """Trainable parameter count."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ----------------------------
# Export to C header
# ----------------------------
def export_c_header(model: nn.Module, sr: int, vocab_size: int, path: str, meta: Dict):
    """
    Exports model weights as a C header for bare metal Daisy inference.
    Inference code will need a matching hand-written C conv forward pass.
    """
    model.eval()

    def ascii_pitch(p: str) -> str:
        # Replace unicode sharp/flat with ASCII equivalents for C strings
        return p.replace("\u266f", "#").replace("\u266d", "b")

    lines = [
        "// Auto-generated by train_onset.py - DO NOT EDIT",
        "// OnsetNet weights for Daisy Seed bare metal inference",
        f"// sr={sr} vocab_size={vocab_size} window_samples={int(round(WINDOW_MS/1000*sr))}",
        "",
        "#pragma once",
        "#include <stdint.h>",
        "",
        f"#define ONSET_SR           {sr}",
        f"#define ONSET_VOCAB_SIZE   {vocab_size}",
        f"#define ONSET_WINDOW       {int(round(WINDOW_MS/1000*sr))}",
        f"#define ONSET_WIDTH        {model.width}",
        "",
        "// Pitch names for each vocab index",
        "static const char* onset_pitch_names[] = {",
    ]
    for p in meta["index_to_pitch"]:
        lines.append(f'    "{ascii_pitch(p)}",')
    lines += ["};", ""]

    def arr(name: str, t: torch.Tensor):
        """Flatten a tensor and write it as a C float array."""
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
        ("conv.0",  "l0_conv"),  # Conv1d
        ("conv.1",  "l0_bn"),    # BN
        ("conv.3",  "l1_conv"),
        ("conv.4",  "l1_bn"),
        ("conv.6",  "l2_conv"),
        ("conv.7",  "l2_bn"),
    ]

    for sd_prefix, c_prefix in layer_map:
        w_key = f"{sd_prefix}.weight"
        if w_key in sd:
            lines += arr(f"{c_prefix}_weight", sd[w_key])

        # BN parameters / buffers
        for suffix in ("bias", "running_mean", "running_var", "weight", "bias"):
            k = f"{sd_prefix}.{suffix}"
            # avoid double-exporting conv weight
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
    """
    Restore training state from checkpoint.
    Returns:
      start_epoch (int): next epoch to run
      best_score (float)
    """
    ckpt = torch.load(resume_path, map_location=device)

    model.load_state_dict(ckpt["model_state"])
    if "optimizer_state" in ckpt and opt is not None:
        opt.load_state_dict(ckpt["optimizer_state"])

    # AMP scaler (if present)
    if "scaler_state" in ckpt and scaler is not None:
        try:
            scaler.load_state_dict(ckpt["scaler_state"])
        except Exception:
            print("[WARN] Could not load scaler_state (ok if switching devices).")

    # Scheduler (optional; if you want exact LR continuation)
    if scheduler is not None and "scheduler_state" in ckpt:
        try:
            scheduler.load_state_dict(ckpt["scheduler_state"])
        except Exception:
            print("[WARN] Could not load scheduler_state (LR schedule may restart).")

    start_epoch = int(ckpt.get("epoch", 0)) + 1
    best_score = float(ckpt.get("best_score", -1.0))

    print(f"[INFO] Resumed from {resume_path}")
    print(f"[INFO] start_epoch={start_epoch} best_score={best_score:.3f}")

    return start_epoch, best_score


# ----------------------------
# Training
# ----------------------------
def train(args):
    """
    Main training loop:
      - build datasets + loaders
      - train for epochs
      - compute val loss + F1 metrics
      - checkpoint best + last
      - export C header on each new best
      - optional LiveNetViz window during training
    """
    set_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={device}")

    # Load dataset metadata + clip list
    clips, sr, vocab_size = load_clips(args.dataset)
    meta = load_json(os.path.join(args.dataset, "metadata.json"))

    # 10ms window length in samples
    W = int(round(WINDOW_MS / 1000.0 * sr))
    print(f"[INFO] clips={len(clips)} sr={sr} vocab={vocab_size} window={W} samples ({WINDOW_MS}ms)")

    # Stage 1: no chords (single notes only)
    if not args.include_chords:
        clips = [c for c in clips if c.clip_type != "chord"]
        print(f"[INFO] Stage1 (no chords): {len(clips)} clips")

    # Split by clip (not by window) to avoid leakage
    train_clips, val_clips = train_val_split(clips, args.val_ratio, args.seed)
    print(f"[INFO] train={len(train_clips)} val={len(val_clips)}")

    # Dataset yields random windows (virtual length)
    train_ds = OnsetDataset(
        train_clips, W,
        p_on=args.p_on, p_neg=args.p_neg,
        virtual_len=args.virtual_len,
        audio_cache_max=args.audio_cache_max,
        preemph_coef=args.preemph_coef,
        noise_std=args.noise_std,
    )
    val_ds = OnsetDataset(
        val_clips, W,
        p_on=args.p_on, p_neg=args.p_neg,
        virtual_len=max(10000, args.virtual_len // 10),
        audio_cache_max=64,
        preemph_coef=args.preemph_coef,
        noise_std=0.0,   # no noise aug at val
    )

    # Loaders: shuffle=False because dataset sampling is already random
    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True,
        persistent_workers=(args.workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=max(1, args.workers // 2), pin_memory=True,
        persistent_workers=(args.workers > 0),
    )

    # Model
    model = OnsetNet(vocab_size=vocab_size, width=args.width).to(device)
    print(f"[INFO] params={model.count_params():,}")

    # pos_weight counteracts 1/vocab class imbalance (esp stage1 single-note)
    pw = torch.full((vocab_size,), float(args.pos_weight), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    # Optimizer + AMP
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = GradScaler(enabled=(device == "cuda"))

    # Cosine LR schedule
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Checkpoint paths
    best_path = os.path.join(args.out, "onset_best.pt")
    last_path = os.path.join(args.out, "onset_last.pt")

    # Default start: fresh run
    start_epoch = 1
    best_score = -1.0

    # Resume (optional)
    if args.resume:
        start_epoch, best_score = try_resume(
            args.resume, model, opt, scaler, scheduler, device
        )

    # Optional visualizer
    viz = None
    if args.viz:
        viz = LiveNetViz(
            model=model,
            meta=meta,
            sr=sr,
            window_samples=W,
            thresh=args.thresh,
            topk=args.viz_topk,
            out_nodes_cap=args.viz_out_nodes,
            weight_viz_every=args.viz_weight_every,
            pulse_speed=args.viz_pulse_speed,
        )
        print("[INFO] LiveNetViz enabled (Matplotlib window should appear).")

    # Infinite iterators so "steps_per_epoch" is decoupled from dataset length
    def infinite(loader):
        while True:
            for b in loader:
                yield b

    train_it = infinite(train_loader)
    val_it   = infinite(val_loader)

    try:
        for ep in range(start_epoch, args.epochs + 1):
            model.train()
            t0 = time.time()
            tr_loss = 0.0

            viz_every = max(1, int(args.viz_every)) if viz is not None else 0

            # ------------------------
            # Train steps
            # ------------------------
            for step in range(args.steps_per_epoch):
                x, y = next(train_it)
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                opt.zero_grad(set_to_none=True)

                with autocast(device_type="cuda", enabled=(device == "cuda")):
                    logits = model(x)
                    loss   = loss_fn(logits, y)

                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()

                tr_loss += float(loss.item())

                # Live visualization update (run a forward pass on one sample)
                if viz is not None and (step % viz_every == 0):
                    viz.update(x_batch=x, y_batch=y, device=device)

            tr_loss /= args.steps_per_epoch
            scheduler.step()

            # ------------------------
            # Validation
            # ------------------------
            model.eval()
            v_loss = 0.0
            all_logits, all_targets = [], []

            with torch.no_grad():
                for _ in range(args.val_steps):
                    x, y = next(val_it)
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                    logits = model(x)
                    v_loss += float(loss_fn(logits, y).item())
                    all_logits.append(logits.cpu())
                    all_targets.append(y.cpu())

            v_loss /= args.val_steps
            all_logits  = torch.cat(all_logits)
            all_targets = torch.cat(all_targets)

            prec, rec, f1 = f1_scores(all_logits, all_targets, thresh=args.thresh)
            lr_now = opt.param_groups[0]["lr"]
            dt = time.time() - t0

            print(
                f"[EP {ep:03d}] train={tr_loss:.4f} val={v_loss:.4f} "
                f"F1={f1:.3f} P={prec:.3f} R={rec:.3f} "
                f"lr={lr_now:.2e} t={dt:.1f}s"
            )

            # ------------------------
            # Checkpoints
            # ------------------------
            # Save LAST checkpoint each epoch (includes scaler/scheduler for resume)
            ckpt = {
                "epoch": ep,
                "model_state": model.state_dict(),
                "optimizer_state": opt.state_dict(),
                "scaler_state": scaler.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_score": best_score,
                "sr": sr,
                "vocab_size": vocab_size,
                "window_samples": W,
                "width": args.width,
                "thresh": args.thresh,
                "args": vars(args),
            }
            torch.save(ckpt, last_path)

            # Save BEST checkpoint + export header on improvement
            if f1 > best_score:
                best_score = f1
                ckpt["best_score"] = best_score
                torch.save(ckpt, best_path)
                print(f"[INFO] New best F1={best_score:.3f} -> {best_path}")

                c_header_path = os.path.join(args.out, "onset_weights.h")
                export_c_header(model, sr, vocab_size, c_header_path, meta)

        print(f"\nDone. Best F1={best_score:.3f}")
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

    # Dataset / output
    ap.add_argument("--dataset",         type=str,   default="dataset")
    ap.add_argument("--out",             type=str,   default="checkpoints")

    # Resume
    ap.add_argument("--resume",          type=str,   default="", help="Path to .pt checkpoint to resume from")

    # Repro + training loop
    ap.add_argument("--seed",            type=int,   default=42)
    ap.add_argument("--epochs",          type=int,   default=80)
    ap.add_argument("--steps_per_epoch", type=int,   default=300)
    ap.add_argument("--val_steps",       type=int,   default=60)
    ap.add_argument("--val_ratio",       type=float, default=0.1)

    # Optimization
    ap.add_argument("--batch",           type=int,   default=2048)
    ap.add_argument("--lr",              type=float, default=1e-3)
    ap.add_argument("--pos_weight",      type=float, default=54.0,  help="BCE pos_weight for class imbalance")

    # Model
    ap.add_argument("--width",           type=int,   default=32, help="Model channel width (32=~650K MACs)")

    # Data loading + sampling
    ap.add_argument("--workers",         type=int,   default=4)
    ap.add_argument("--audio_cache_max", type=int,   default=256)
    ap.add_argument("--virtual_len",     type=int,   default=100000)

    # Threshold for metrics + viz line
    ap.add_argument("--thresh",          type=float, default=0.5)

    # Sampling ratios
    ap.add_argument("--p_on",            type=float, default=0.6, help="Fraction of transient windows")
    ap.add_argument("--p_neg",           type=float, default=0.4, help="Fraction of silence windows")

    # Augmentations
    ap.add_argument("--preemph_coef",    type=float, default=0.97, help="Pre-emphasis (0=off)")
    ap.add_argument("--noise_std",       type=float, default=0.001, help="Additive noise std")

    # Stage selection
    ap.add_argument("--include_chords",  action="store_true", help="Include chord clips (stage 2)")

    # Live viz flags
    ap.add_argument("--viz",             action="store_true", help="Enable live wiring-diagram visualizer")
    ap.add_argument("--viz_every",       type=int, default=30, help="Update viz every N train steps")
    ap.add_argument("--viz_topk",        type=int, default=8,  help="Top-K notes to show in prediction panel")
    ap.add_argument("--viz_out_nodes",   type=int, default=55, help="How many output nodes to draw (cap)")

    # Animation/perf knobs
    ap.add_argument("--viz_weight_every", type=int, default=2,
                    help="Update weight-colored edges every N viz updates (higher = faster training)")
    ap.add_argument("--viz_pulse_speed",  type=float, default=0.35,
                    help="Pulse speed along the chosen path (higher = faster animation)")

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()