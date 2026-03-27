# export_torchscript.py
import torch
from model import DDSPGuitarToPiano, FRAME_SIZE

ckpt_path = "./checkpoints/best_model.pt"
out_path = "./checkpoints/guitar_to_piano_ts.pt"

device = "cpu"

model = DDSPGuitarToPiano()
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
model.eval()

example = torch.randn(1, FRAME_SIZE)

# If your model has infer_frame(), wrap that behavior in forward() first,
# or export a wrapper module that calls infer_frame().
class Wrapper(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        if hasattr(self.m, "infer_frame"):
            return self.m.infer_frame(x)
        y = self.m(x)
        return y[0] if isinstance(y, (tuple, list)) else y

wrapper = Wrapper(model).eval()

with torch.no_grad():
    ts = torch.jit.trace(wrapper, example)

ts.save(out_path)
print(f"Saved TorchScript model to {out_path}")