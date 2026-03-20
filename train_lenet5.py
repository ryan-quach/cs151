"""
train_lenet5.py  —  Train LeNet-5 on MNIST, quantize to int8, export hex files
================================================================================
Requirements:
    pip install torch torchvision numpy

Usage:
    python train_lenet5.py

Outputs (written to ./weights/):
    c1_w.hex  c1_b.hex
    c3_w.hex  c3_b.hex
    c5_w_0.hex  c5_w_1.hex  c5_b.hex   (c5 weights split to stay under
                                         EDA Playground's 100k-char limit)
    f6_w.hex  f6_b.hex
    out_w.hex  out_b.hex
    test_image.hex          32×32 padded MNIST digit
    test_image_label.txt    ground-truth label

At the end the script prints:
  - The max accumulator value seen at each layer (before right-shift)
  - The recommended RSHIFT value for each layer
  - A ready-to-paste localparam block for lenet_top in lenet.v

Why RSHIFT is needed
--------------------
int8 weights are scaled to fill [-127, 127] (scale ≈ 200–400×).
After N MACs the 24-bit accumulator holds values ≈ N × 127 × 127,
which far exceeds the [0,127] ReLU range.  A right-shift of RSHIFT
bits after the accumulation (before the clamp) scales the result
back into the representable range.  The correct value depends on the
trained weights and is computed here by a Python integer simulation
that exactly mirrors the RTL arithmetic.
"""

import math, os, struct
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from numpy.lib.stride_tricks import sliding_window_view

# ── Reproducibility ───────────────────────────────────────────────────────────
torch.manual_seed(42)
np.random.seed(42)

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS     = 5
BATCH_SIZE = 64
LR         = 1e-3
OUT_DIR    = "weights"
TEST_DIGIT = 7

os.makedirs(OUT_DIR, exist_ok=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_ds = datasets.MNIST("./data", train=True,  download=True, transform=transform)
test_ds  = datasets.MNIST("./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=1000,       shuffle=False)

# ── Model ─────────────────────────────────────────────────────────────────────
class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1  = nn.Conv2d(1,   6,   5)
        self.s2  = nn.MaxPool2d(2, 2)
        self.c3  = nn.Conv2d(6,   16,  5)
        self.s4  = nn.MaxPool2d(2, 2)
        self.c5  = nn.Conv2d(16,  120, 5)
        self.f6  = nn.Linear(120, 84)
        self.out = nn.Linear(84,  10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.s2(x)
        x = self.relu(self.c3(x))
        x = self.s4(x)
        x = self.relu(self.c5(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.f6(x))
        return self.out(x)

model     = LeNet5().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ── Training ──────────────────────────────────────────────────────────────────
print("Training LeNet-5 on MNIST...")
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_loader)

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            preds = model(imgs).argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
    print(f"  Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={100.*correct/total:.2f}%")

# ── Quantisation helpers ──────────────────────────────────────────────────────
def quantize_to_int8(tensor):
    max_abs = np.max(np.abs(tensor))
    if max_abs == 0:
        return np.zeros_like(tensor, dtype=np.int8), 1.0
    scale = 127.0 / max_abs
    return np.clip(np.round(tensor * scale), -128, 127).astype(np.int8), scale

def write_hex(array, path):
    with open(path, "w") as f:
        for v in array.flatten():
            byte = struct.pack("b", int(v))[0]
            f.write(f"{byte:02X}\n")
    print(f"  Wrote {path}  ({array.size} values)")

def get_np(p):
    return p.detach().cpu().numpy()

# ── Extract and quantise weights ─────────────────────────────────────────────
print("\nQuantizing weights to int8...")

layers = {
    "c1":  (model.c1.weight,  model.c1.bias),
    "c3":  (model.c3.weight,  model.c3.bias),
    "c5":  (model.c5.weight,  model.c5.bias),
    "f6":  (model.f6.weight,  model.f6.bias),
    "out": (model.out.weight, model.out.bias),
}

quantized = {}
for name, (w_param, b_param) in layers.items():
    w_np = get_np(w_param)
    b_np = get_np(b_param)
    w_q, scale = quantize_to_int8(w_np)
    b_q = np.clip(np.round(b_np * scale), -128, 127).astype(np.int8)
    quantized[name] = (w_q, b_q, scale)

    print(f"\n  [{name}]  float range=[{w_np.min():.4f}, {w_np.max():.4f}]  scale={scale:.2f}")
    print(f"        weight shape: {w_np.shape}  -> {w_q.size} int8 values")

    if name == "c5":
        flat = w_q.flatten()
        half = len(flat) // 2
        write_hex(flat[:half], os.path.join(OUT_DIR, "c5_w_0.hex"))
        write_hex(flat[half:], os.path.join(OUT_DIR, "c5_w_1.hex"))
        print(f"  (c5 weights split: c5_w_0.hex [0..{half-1}], c5_w_1.hex [{half}..{len(flat)-1}])")
    else:
        write_hex(w_q, os.path.join(OUT_DIR, f"{name}_w.hex"))
    write_hex(b_q, os.path.join(OUT_DIR, f"{name}_b.hex"))

# ── Verify counts ─────────────────────────────────────────────────────────────
expected = {"c1":(150,6), "c3":(2400,16), "c5":(48000,120), "f6":(10080,84), "out":(840,10)}
print("\nVerifying weight counts:")
for name, (we, be) in expected.items():
    w_q, b_q, _ = quantized[name]
    print(f"  [{name}]  weights={w_q.size}/{we} {'OK' if w_q.size==we else 'FAIL'}"
          f"  biases={b_q.size}/{be} {'OK' if b_q.size==be else 'FAIL'}")

# ── Export test image ─────────────────────────────────────────────────────────
print(f"\nExporting test image (digit class = {TEST_DIGIT})...")
test_img_tensor = None
for imgs, labels in DataLoader(test_ds, batch_size=1, shuffle=False):
    if labels.item() == TEST_DIGIT:
        test_img_tensor = imgs
        break

model.eval()
with torch.no_grad():
    pred = model(test_img_tensor.to(DEVICE)).argmax(dim=1).item()
print(f"  Float model prediction: {pred}  (ground truth: {TEST_DIGIT})")

mean, std = 0.1307, 0.3081
img_np    = test_img_tensor.squeeze().numpy()
img_int8  = np.clip((img_np * std + mean) * 127, 0, 127).astype(np.int8)

write_hex(img_int8, os.path.join(OUT_DIR, "test_image.hex"))
with open(os.path.join(OUT_DIR, "test_image_label.txt"), "w") as f:
    f.write(f"{TEST_DIGIT}\n")
print(f"  Wrote test_image_label.txt  (ground truth = {TEST_DIGIT})")

# ── Integer forward-pass simulation  (mirrors RTL exactly) ───────────────────
print("\nSimulating integer forward pass to compute RSHIFT values...")

w_c1,  b_c1,  _ = quantized["c1"]
w_c3,  b_c3,  _ = quantized["c3"]
w_c5,  b_c5,  _ = quantized["c5"]
w_f6,  b_f6,  _ = quantized["f6"]
w_out, b_out, _ = quantized["out"]

def int_conv_layer(x, w, b, rshift):
    """
    Exact integer conv matching the RTL:
      acc = bias + sum(w_int8 * x_int8)
      out = clip(acc >> rshift, 0, 127)
    x:  (in_ch, H, W)  int8
    w:  (out_ch, in_ch, kH, kW)  int8
    b:  (out_ch,)  int8
    Returns: (out_ch, oH, oW) int8,  scalar max |acc| before shift
    """
    oc, ic, kH, kW = w.shape
    H, W = x.shape[1], x.shape[2]
    oH, oW = H - kH + 1, W - kW + 1

    x64 = x.astype(np.int64)
    w64 = w.astype(np.int64)
    accs = np.zeros((oc, oH, oW), dtype=np.int64)

    for c in range(ic):
        wins = sliding_window_view(x64[c], (kH, kW))   # (oH, oW, kH, kW)
        for o in range(oc):
            accs[o] += (wins * w64[o, c]).sum(axis=(-2, -1))

    for o in range(oc):
        accs[o] += int(b[o])

    max_abs = int(np.max(np.abs(accs)))
    out = np.clip(accs >> rshift, 0, 127).astype(np.int8)
    return out, max_abs

def int_maxpool(x):
    ch, H, W = x.shape
    oH, oW = H // 2, W // 2
    out = np.zeros((ch, oH, oW), dtype=np.int8)
    for c in range(ch):
        out[c] = np.maximum(
            np.maximum(x[c, 0::2, 0::2], x[c, 0::2, 1::2]),
            np.maximum(x[c, 1::2, 0::2], x[c, 1::2, 1::2])
        )
    return out

def int_fc_layer(x, w_flat, b, rshift, use_relu):
    """
    Exact integer FC matching the RTL.
    x:      (in_size,)  int8
    w_flat: (out_size * in_size,)  int8 row-major
    b:      (out_size,)  int8
    """
    out_size = len(b)
    in_size  = len(x)
    w2d  = w_flat.reshape(out_size, in_size).astype(np.int64)
    accs = w2d @ x.astype(np.int64) + b.astype(np.int64)
    max_abs = int(np.max(np.abs(accs)))
    shifted = accs >> rshift
    if use_relu:
        out = np.clip(shifted, 0, 127).astype(np.int8)
    else:
        out = np.clip(shifted, -128, 127).astype(np.int8)
    return out, max_abs

def rshift_for(max_acc):
    """Minimum RSHIFT so that max_acc >> rshift <= 127."""
    if max_acc <= 127:
        return 0
    return math.floor(math.log2(max_acc / 127))

# Pass 1: run with RSHIFT=0 to measure actual max accumulations
x = img_int8.reshape(1, 32, 32)

out_c1, max_c1 = int_conv_layer(x,      w_c1.reshape(6,1,5,5),   b_c1, 0)
out_s2          = int_maxpool(out_c1)
out_c3, max_c3 = int_conv_layer(out_s2, w_c3.reshape(16,6,5,5),  b_c3, 0)
out_s4          = int_maxpool(out_c3)
out_c5, max_c5 = int_conv_layer(out_s4, w_c5.reshape(120,16,5,5),b_c5, 0)
out_f6, max_f6 = int_fc_layer(out_c5.flatten(), w_f6.flatten(),   b_f6, 0, True)
_,      max_ot = int_fc_layer(out_f6,           w_out.flatten(),  b_out, 0, False)

rs_c1  = rshift_for(max_c1)
rs_c3  = rshift_for(max_c3)
rs_c5  = rshift_for(max_c5)
rs_f6  = rshift_for(max_f6)
rs_out = rshift_for(max_ot)

print(f"  Layer  max|acc| before shift  recommended RSHIFT")
print(f"  -----  ----------------------  ------------------")
print(f"  C1     {max_c1:>20,d}  {rs_c1}")
print(f"  C3     {max_c3:>20,d}  {rs_c3}")
print(f"  C5     {max_c5:>20,d}  {rs_c5}")
print(f"  F6     {max_f6:>20,d}  {rs_f6}")
print(f"  OUT    {max_ot:>20,d}  {rs_out}")

# Pass 2: run with computed RSHIFTs and verify prediction
out_c1, _ = int_conv_layer(x,      w_c1.reshape(6,1,5,5),   b_c1, rs_c1)
out_s2     = int_maxpool(out_c1)
out_c3, _ = int_conv_layer(out_s2, w_c3.reshape(16,6,5,5),  b_c3, rs_c3)
out_s4     = int_maxpool(out_c3)
out_c5, _ = int_conv_layer(out_s4, w_c5.reshape(120,16,5,5),b_c5, rs_c5)
out_f6, _ = int_fc_layer(out_c5.flatten(), w_f6.flatten(),   b_f6, rs_f6,  True)
out_ot, _ = int_fc_layer(out_f6,           w_out.flatten(),  b_out, rs_out, False)

int_pred = int(np.argmax(out_ot))
print(f"\n  Integer simulation prediction: {int_pred}  (ground truth: {TEST_DIGIT})")
if int_pred == TEST_DIGIT:
    print("  Simulation CORRECT — these RSHIFT values should work in hardware.")
else:
    print("  Simulation WRONG — try increasing RSHIFT values by 1 and re-check.")

print(f"""
================================================================================
Paste these RSHIFT values into lenet_top in lenet.v:

    localparam RSHIFT_C1  = {rs_c1};
    localparam RSHIFT_C3  = {rs_c3};
    localparam RSHIFT_C5  = {rs_c5};
    localparam RSHIFT_F6  = {rs_f6};
    localparam RSHIFT_OUT = {rs_out};

Then in EDA Playground:
  1. Replace lenet.v with the updated file (includes RSHIFT parameter support)
  2. Update the five localparam lines above with these values
  3. Re-upload hex files if you changed them
  4. Set TRAINED_WEIGHTS=1 and EXPECTED_CLASS={TEST_DIGIT} in the testbench
  5. Run — the system test should show: Classification : CORRECT
================================================================================
""")