Jetson Orin deployment
======================

Copy this folder to the Jetson, then run:

  python3 jetson_realtime.py --model ./model_pruned_scripted.pt --input ./guitar.wav --play

Optional save while playing:

  python3 jetson_realtime.py --model ./model_pruned_scripted.pt --input ./guitar.wav --play --output ./piano_out.wav

For live input instead of WAV:

  python3 jetson_realtime.py --model ./model_pruned_scripted.pt --live

Useful Jetson performance settings:

  sudo nvpmodel -m 0
  sudo jetson_clocks

If sounddevice cannot find the right output, run:

  python3 jetson_realtime.py --list-devices

Then pass --output_device INDEX.
