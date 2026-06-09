# Jetson Orin pruning/export/deployment

## 1. Export on your training machine

Use the trained 512-window checkpoint from this patched project. Do **not** use an older 1024-window checkpoint unless the architecture constants match.

```bash
python prune_export_jetson.py \
  --checkpoint ./checkpoints/best_model.pt \
  --output_dir ./jetson_export \
  --prune_amount 0.35 \
  --device cuda
```

This creates:

- `jetson_export/model_pruned_scripted.pt`
- `jetson_export/model_pruned_checkpoint.pt`
- `jetson_export/jetson_realtime.py`
- `jetson_export/model.py`
- `jetson_export/jetson_deploy_bundle.tar.gz`

## 2. Copy to Jetson

```bash
scp jetson_export/jetson_deploy_bundle.tar.gz jetson@JETSON_IP:~/
ssh jetson@JETSON_IP
mkdir -p ~/piano_jetson
mv ~/jetson_deploy_bundle.tar.gz ~/piano_jetson/
cd ~/piano_jetson
tar -xzf jetson_deploy_bundle.tar.gz
```

## 3. Run playback like the old `--play` mode

```bash
python3 jetson_realtime.py \
  --model ./model_pruned_scripted.pt \
  --input ./guitar.wav \
  --play
```

Save while playing:

```bash
python3 jetson_realtime.py \
  --model ./model_pruned_scripted.pt \
  --input ./guitar.wav \
  --play \
  --output ./piano_out.wav
```

Live input:

```bash
python3 jetson_realtime.py --model ./model_pruned_scripted.pt --live
```

List devices:

```bash
python3 jetson_realtime.py --list-devices
```

Then pass device indexes if needed:

```bash
python3 jetson_realtime.py \
  --model ./model_pruned_scripted.pt \
  --input ./guitar.wav \
  --play \
  --output_device 0
```

## Notes

- Unstructured pruning helps compression and can regularize the model, but dense PyTorch CUDA kernels may not get much faster from zeros alone.
- For real Jetson speed, the biggest wins are: the 512-frame model, TorchScript, CUDA, `torch.inference_mode()`, and the normalized overlap-add engine.
- If pruning hurts piano quality, try `--prune_amount 0.15` or `0.25`.
- If quality survives, try `0.45`. I would not start above `0.50` for audio.
- Keep output `--volume` around `0.8` to `0.9` while testing distortion.

Optional Jetson performance mode:

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```
