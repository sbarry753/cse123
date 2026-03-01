"""
export_model_json.py

Exports an OnsetNet checkpoint to JSON for the browser synth.

Usage:
    python export_model_json.py --ckpt stage1_5ms_cos_sus/onset_best.pt \
                                --meta labels/metadata.json \
                                --out model.json

The JSON will contain:
  - state_dict: all weight arrays as nested lists (PyTorch key names)
  - vocab_size, width, pitch_names, midi_for_idx
"""

import json
import argparse
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True, help='Path to .pt checkpoint')
    ap.add_argument('--meta', default='', help='Path to metadata.json (for pitch names)')
    ap.add_argument('--out', default='model.json', help='Output JSON path')
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    sd = ckpt['model_state']
    vocab_size = int(ckpt.get('vocab_size', 0))
    width = int(ckpt.get('width', 96))

    print(f"[INFO] vocab_size={vocab_size}  width={width}")
    print(f"[INFO] state_dict keys: {list(sd.keys())}")

    # Load pitch names from metadata if available
    pitch_names = None
    midi_for_idx = None
    if args.meta:
        try:
            with open(args.meta) as f:
                meta = json.load(f)
            pitch_names = meta.get('index_to_pitch', None)
            print(f"[INFO] pitch_names ({len(pitch_names)}): {pitch_names[:8]}...")
        except Exception as e:
            print(f"[WARN] Could not load meta: {e}")

    # Build MIDI index for each vocab entry
    # Guitar range E2=40..B5=83 — map pitch name to midi number
    NOTE_SEMITONES = {'C':0,'C#':1,'Db':1,'D':2,'D#':3,'Eb':3,'E':4,'F':5,
                      'F#':6,'Gb':6,'G':7,'G#':8,'Ab':8,'A':9,'A#':10,'Bb':10,'B':11}
    def pitch_to_midi(name):
        # e.g. "E2", "C#3", "Bb4"
        name = name.replace('\u266f','#').replace('\u266d','b')
        for acc in ['#b', '#', 'b', '']:
            for note, sem in NOTE_SEMITONES.items():
                if name.startswith(note):
                    rest = name[len(note):]
                    try:
                        octave = int(rest)
                        return (octave + 1) * 12 + sem
                    except:
                        pass
        return 60  # fallback

    if pitch_names:
        midi_for_idx = [pitch_to_midi(p) for p in pitch_names]
        print(f"[INFO] midi_for_idx sample: {list(zip(pitch_names[:4], midi_for_idx[:4]))}")

    # Serialise all weights as nested Python lists
    # The browser loader reads keys exactly as PyTorch names them:
    # conv.0.weight, conv.1.weight (BN gamma), conv.1.bias (BN beta),
    # conv.1.running_mean, conv.1.running_var, ... etc.
    state_dict_json = {}
    for k, v in sd.items():
        arr = v.float().numpy()
        state_dict_json[k] = arr.tolist()
        print(f"  {k}: {list(arr.shape)}")

    out = {
        'state_dict': state_dict_json,
        'vocab_size': vocab_size,
        'width': width,
        'pitch_names': pitch_names,
        'midi_for_idx': midi_for_idx,
    }

    with open(args.out, 'w') as f:
        json.dump(out, f)

    import os
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"\n[OK] Saved to {args.out}  ({size_mb:.1f} MB)")
    print("Load this file in the synth using the 'Load Model JSON' button.")

if __name__ == '__main__':
    main()