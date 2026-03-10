from pathlib import Path
root = Path(r"\data\GuitarSet")
print("root contents:", [p.name for p in root.iterdir()])
# check for nested subfolder
for p in root.rglob("audio_hex-pickup_debleeded"):
    print("found hex dir at:", p)