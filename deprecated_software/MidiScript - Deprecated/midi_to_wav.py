#!/usr/bin/env python3
"""
midi_split.py — Split a MIDI file into per-instrument WAV files.

Requirements:
    pip install mido pretty_midi fluidsynth

You also need a General MIDI soundfont, e.g.:
    apt-get install -y fluid-soundfont-gm
The default soundfont path below is for that package.
"""

import sys
import os
import pretty_midi
import fluidsynth
import numpy as np
import wave
import struct

SOUNDFONT = "/usr/share/sounds/sf2/FluidR3_GM.sf2"
SAMPLE_RATE = 44100

# General MIDI program number → friendly name
def program_to_name(program, is_drum=False):
    if is_drum:
        return "Drums"
    gm_names = [
        "Acoustic_Grand_Piano", "Bright_Acoustic_Piano", "Electric_Grand_Piano",
        "Honky_Tonk_Piano", "Electric_Piano_1", "Electric_Piano_2", "Harpsichord",
        "Clavinet", "Celesta", "Glockenspiel", "Music_Box", "Vibraphone",
        "Marimba", "Xylophone", "Tubular_Bells", "Dulcimer", "Drawbar_Organ",
        "Percussive_Organ", "Rock_Organ", "Church_Organ", "Reed_Organ", "Accordion",
        "Harmonica", "Tango_Accordion", "Acoustic_Guitar_Nylon", "Acoustic_Guitar_Steel",
        "Electric_Guitar_Jazz", "Electric_Guitar_Clean", "Electric_Guitar_Muted",
        "Overdriven_Guitar", "Distortion_Guitar", "Guitar_Harmonics",
        "Acoustic_Bass", "Electric_Bass_Finger", "Electric_Bass_Pick", "Fretless_Bass",
        "Slap_Bass_1", "Slap_Bass_2", "Synth_Bass_1", "Synth_Bass_2",
        "Violin", "Viola", "Cello", "Contrabass", "Tremolo_Strings",
        "Pizzicato_Strings", "Orchestral_Harp", "Timpani",
        "String_Ensemble_1", "String_Ensemble_2", "Synth_Strings_1", "Synth_Strings_2",
        "Choir_Aahs", "Voice_Oohs", "Synth_Voice", "Orchestra_Hit",
        "Trumpet", "Trombone", "Tuba", "Muted_Trumpet", "French_Horn",
        "Brass_Section", "Synth_Brass_1", "Synth_Brass_2",
        "Soprano_Sax", "Alto_Sax", "Tenor_Sax", "Baritone_Sax",
        "Oboe", "English_Horn", "Bassoon", "Clarinet",
        "Piccolo", "Flute", "Recorder", "Pan_Flute", "Blown_Bottle",
        "Shakuhachi", "Whistle", "Ocarina",
        "Lead_Square", "Lead_Sawtooth", "Lead_Calliope", "Lead_Chiff",
        "Lead_Charang", "Lead_Voice", "Lead_Fifths", "Lead_Bass_Lead",
        "Pad_New_Age", "Pad_Warm", "Pad_Polysynth", "Pad_Choir",
        "Pad_Bowed", "Pad_Metallic", "Pad_Halo", "Pad_Sweep",
        "FX_Rain", "FX_Soundtrack", "FX_Crystal", "FX_Atmosphere",
        "FX_Brightness", "FX_Goblins", "FX_Echoes", "FX_Sci_Fi",
        "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba", "Bagpipe",
        "Fiddle", "Shanai",
        "Tinkle_Bell", "Agogo", "Steel_Drums", "Woodblock", "Taiko_Drum",
        "Melodic_Tom", "Synth_Drum", "Reverse_Cymbal",
        "Guitar_Fret_Noise", "Breath_Noise", "Seashore", "Bird_Tweet",
        "Telephone_Ring", "Helicopter", "Applause", "Gunshot",
    ]
    if 0 <= program < len(gm_names):
        return gm_names[program]
    return f"Program_{program}"

def render_instrument_to_wav(midi_path, instrument, out_path):
    # Build a new PrettyMIDI with only this instrument
    new_midi = pretty_midi.PrettyMIDI()
    new_midi.instruments.append(instrument)

    # Synthesize to audio
    audio = new_midi.fluidsynth(fs=SAMPLE_RATE, sf2_path=SOUNDFONT)

    # Convert float32 [-1, 1] to int16
    audio_int16 = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_int16 * 32767).astype(np.int16)

    # Write WAV
    with wave.open(out_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    print(f"  Written: {out_path}")

def split_midi(midi_path):
    if not os.path.exists(midi_path):
        print(f"Error: file not found: {midi_path}")
        sys.exit(1)

    base = os.path.splitext(midi_path)[0]
    midi = pretty_midi.PrettyMIDI(midi_path)

    print(f"Found {len(midi.instruments)} instrument track(s) in {midi_path}")

    name_counts = {}
    for instrument in midi.instruments:
        name = program_to_name(instrument.program, instrument.is_drum)

        # Handle duplicate instrument names (e.g. two piano tracks)
        count = name_counts.get(name, 0)
        name_counts[name] = count + 1
        suffix = f"_{count + 1}" if count > 0 else ""

        out_path = f"{base}_{name}{suffix}.wav"
        print(f"  Rendering: {name}{suffix}...")
        render_instrument_to_wav(midi_path, instrument, out_path)

    print("Done!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 midi_split.py <file.midi> [file2.midi ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        split_midi(path)
