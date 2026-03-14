"""
inspect_transients.py

Plots the first 10ms of each note from scale recordings,
overlaid by pitch class so you can see if transients are distinct.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from pathlib import Path

DATASET_ROOT = "labels"
MS_TO_SHOW = 10.0

def load_manifest(root):
    items = []
    with open(os.path.join(root, "manifest.jsonl")) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def load_label(root, item):
    with open(os.path.join(root, item["label"])) as f:
        return json.load(f)

def main():
    meta_path = os.path.join(DATASET_ROOT, "metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    sr = meta["sr"]
    index_to_pitch = meta["index_to_pitch"]

    manifest = load_manifest(DATASET_ROOT)

    # Only scale segments
    scale_items = [it for it in manifest if it["type"] == "scale_segment"]
    print(f"Found {len(scale_items)} scale segments")

    window = int((MS_TO_SHOW / 1000.0) * sr)

    # Group by midi note
    from collections import defaultdict
    by_midi = defaultdict(list)

    for it in scale_items:
        lab = load_label(DATASET_ROOT, it)
        audio_path = os.path.join(DATASET_ROOT, it["audio"])
        y, _ = sf.read(audio_path, dtype="float32", always_2d=False)

        onset = int(lab["onset_sample"])
        t_start = int(lab["transient_window_start"])

        # Grab 10ms from onset
        seg = y[onset:onset + window]
        if len(seg) < window:
            seg = np.pad(seg, (0, window - len(seg)))

        midi = lab["active_notes_midi"][0]
        by_midi[midi].append(seg)

    # Plot: one subplot per octave grouping
    all_midis = sorted(by_midi.keys())
    n = len(all_midis)
    cols = 6
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(18, rows * 2.5))
    axes = axes.flatten()

    t_ms = np.linspace(0, MS_TO_SHOW, window)

    for i, midi in enumerate(all_midis):
        ax = axes[i]
        segs = by_midi[midi]
        pitch = librosa.midi_to_note(midi)

        for seg in segs:
            ax.plot(t_ms, seg, alpha=0.6, linewidth=0.8)

        ax.set_title(f"{pitch} (midi {midi})\nn={len(segs)}", fontsize=8)
        ax.set_xlim(0, MS_TO_SHOW)
        ax.axvline(x=0, color='r', linewidth=0.5)
        ax.set_xlabel("ms", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("First 10ms transient per note (all scale recordings)", fontsize=12)
    plt.tight_layout()
    plt.savefig("transient_inspection.png", dpi=150)
    plt.show()
    print("Saved transient_inspection.png")

    # Also plot overlaid comparison of adjacent semitones
    # Pick a few pairs that are hardest to distinguish
    print("\nPlotting adjacent semitone comparison...")
    fig2, axes2 = plt.subplots(2, 4, figsize=(16, 6))
    axes2 = axes2.flatten()

    pairs = [(all_midis[i], all_midis[i+1]) for i in range(0, min(8, len(all_midis)-1))]
    for i, (m1, m2) in enumerate(pairs):
        ax = axes2[i]
        p1 = librosa.midi_to_note(m1)
        p2 = librosa.midi_to_note(m2)

        for seg in by_midi[m1][:3]:
            ax.plot(t_ms, seg, color='blue', alpha=0.7, linewidth=0.8, label=p1)
        for seg in by_midi[m2][:3]:
            ax.plot(t_ms, seg, color='red', alpha=0.7, linewidth=0.8, label=p2)

        ax.set_title(f"{p1} vs {p2}", fontsize=9)
        ax.set_xlim(0, MS_TO_SHOW)
        handles = [
            plt.Line2D([0], [0], color='blue', label=p1),
            plt.Line2D([0], [0], color='red', label=p2),
        ]
        ax.legend(handles=handles, fontsize=7)
        ax.tick_params(labelsize=6)

    plt.suptitle("Adjacent semitone transient comparison", fontsize=12)
    plt.tight_layout()
    plt.savefig("semitone_comparison.png", dpi=150)
    plt.show()
    print("Saved semitone_comparison.png")

if __name__ == "__main__":
    main()