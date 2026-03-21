"""
benchmark.py  —  CPU vs GPU inference speed comparison for LeNet-5
===================================================================
Run this after train_lenet5.py has already trained and saved the model.
If train_lenet5.py hasn't been run yet, this script will train first.

Usage:
    python benchmark.py

Prints a comparison table of:
  - CPU inference (single image, batch)
  - GPU inference (single image, batch)  [if CUDA available]
  - Hardware FPGA theoretical numbers    (from RTL analysis)

Requirements:
    pip install torch torchvision numpy
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# ── Model definition (must match train_lenet5.py) ────────────────────────────
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

# ── Load or train model ───────────────────────────────────────────────────────
torch.manual_seed(42)

transform = transforms.Compose([
    transforms.Pad(2),
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
test_ds = datasets.MNIST("./data", train=False, download=True, transform=transform)

model = LeNet5()
try:
    model.load_state_dict(torch.load("lenet5_weights.pth", map_location="cpu"))
    print("Loaded saved model weights.")
except FileNotFoundError:
    print("No saved weights found — training for 5 epochs...")
    from torchvision import datasets as ds2
    train_ds = ds2.MNIST("./data", train=True, download=True, transform=transform)
    loader   = DataLoader(train_ds, batch_size=64, shuffle=True)
    opt      = torch.optim.Adam(model.parameters(), lr=1e-3)
    crit     = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(5):
        for imgs, labels in loader:
            opt.zero_grad()
            crit(model(imgs), labels).backward()
            opt.step()
        print(f"  Epoch {epoch+1}/5 done")
    torch.save(model.state_dict(), "lenet5_weights.pth")
    print("Saved weights to lenet5_weights.pth")

model.eval()

# ── Benchmark helper ──────────────────────────────────────────────────────────
WARMUP_RUNS  = 50
TIMED_RUNS   = 500
BATCH_SIZES  = [1, 32, 256, 1024]

def benchmark(device_name, batch_size, n_runs, warmup):
    device = torch.device(device_name)
    m = LeNet5().to(device)
    m.load_state_dict(model.state_dict())
    m.eval()

    dummy = torch.randn(batch_size, 1, 32, 32, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = m(dummy)

    if device_name == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = m(dummy)
            if device_name == "cuda":
                torch.cuda.synchronize()
    end = time.perf_counter()

    elapsed_s      = end - start
    total_images   = n_runs * batch_size
    throughput     = total_images / elapsed_s
    latency_ms     = (elapsed_s / n_runs) * 1000   # per batch
    latency_per_img = latency_ms / batch_size

    return throughput, latency_ms, latency_per_img

# ── Hardware theoretical numbers ──────────────────────────────────────────────
HW_SEQ_CYCLES   = 418_096
HW_PAR_CYCLES   =  35_425
HW_PIPE_CYCLES  =  19_600   # bottleneck for pipelined
HW_CLK_MHZ      = 100
HW_SEQ_THRU     = HW_CLK_MHZ * 1e6 / HW_SEQ_CYCLES
HW_PAR_THRU     = HW_CLK_MHZ * 1e6 / HW_PAR_CYCLES
HW_PIPE_THRU    = HW_CLK_MHZ * 1e6 / HW_PIPE_CYCLES
HW_SEQ_LAT_MS   = HW_SEQ_CYCLES / (HW_CLK_MHZ * 1e6) * 1000
HW_PAR_LAT_MS   = HW_PAR_CYCLES / (HW_CLK_MHZ * 1e6) * 1000
HW_POWER_W      = 0.5   # typical FPGA dynamic power for this design (est.)

# ── Run benchmarks ────────────────────────────────────────────────────────────
devices = ["cpu"]
if torch.cuda.is_available():
    devices.append("cuda")
    gpu_name = torch.cuda.get_device_name(0)
else:
    gpu_name = "N/A (not available)"

print("\n" + "="*70)
print("  LeNet-5 Inference Benchmark")
print("="*70)
print(f"  CPU : {torch.get_num_threads()} threads")
print(f"  GPU : {gpu_name}")
print(f"  Runs per config: {TIMED_RUNS} (+ {WARMUP_RUNS} warmup)")
print("="*70)

results = {}
for device in devices:
    results[device] = {}
    print(f"\n  Benchmarking {device.upper()}...")
    for bs in BATCH_SIZES:
        thru, lat_ms, lat_per_img = benchmark(device, bs, TIMED_RUNS, WARMUP_RUNS)
        results[device][bs] = (thru, lat_ms, lat_per_img)
        print(f"    batch={bs:>4}  throughput={thru:>10,.0f} img/s  "
              f"latency={lat_ms:>8.3f} ms/batch  ({lat_per_img*1000:>6.2f} us/img)")

# ── Summary comparison table ──────────────────────────────────────────────────
print("\n" + "="*70)
print("  COMPARISON SUMMARY  (single-image latency + peak throughput)")
print("="*70)
print(f"  {'Platform':<35}  {'Latency (1 img)':>16}  {'Throughput':>14}")
print(f"  {'-'*35}  {'-'*16}  {'-'*14}")

# CPU single image
cpu_lat = results["cpu"][1][2]   # latency per image at batch=1
cpu_thru = results["cpu"][max(BATCH_SIZES)][0]
print(f"  {'CPU (PyTorch float32, batch=1)':<35}  {cpu_lat*1000:>13.3f} us  {cpu_thru:>12,.0f}/s")

# GPU single image
if "cuda" in results:
    gpu_lat = results["cuda"][1][2]
    gpu_thru = results["cuda"][max(BATCH_SIZES)][0]
    print(f"  {f'GPU ({gpu_name[:20]}, batch=1)':<35}  {gpu_lat*1000:>13.3f} us  {gpu_thru:>12,.0f}/s")
else:
    print(f"  {'GPU (not available)':<35}  {'—':>16}  {'—':>14}")

print(f"  {'-'*35}  {'-'*16}  {'-'*14}")
print(f"  {'FPGA sequential (lenet.v)':<35}  {HW_SEQ_LAT_MS*1000:>13.0f} us  {HW_SEQ_THRU:>12,.0f}/s")
print(f"  {'FPGA parallel (lenet_parallel.v)':<35}  {HW_PAR_LAT_MS*1000:>13.0f} us  {HW_PAR_THRU:>12,.0f}/s")
print(f"  {'FPGA parallel + pipelined':<35}  {HW_PAR_LAT_MS*1000:>13.0f} us  {HW_PIPE_THRU:>12,.0f}/s")

print("""
  Notes:
  - CPU/GPU use 32-bit float; FPGA uses 8-bit integer (quantized)
  - GPU latency at batch=1 is dominated by kernel launch overhead
  - GPU throughput advantage grows with batch size
  - FPGA latency is deterministic; CPU/GPU latency varies
  - FPGA estimated power: ~0.5W dynamic; GPU: 200-350W TDP
  - FPGA throughput does not depend on batch size (streaming)
""")

# ── Save results for analysis.md ─────────────────────────────────────────────
with open("benchmark_results.txt", "w") as f:
    f.write("LeNet-5 Benchmark Results\n")
    f.write("="*50 + "\n\n")
    for device in devices:
        f.write(f"{device.upper()} Results:\n")
        for bs in BATCH_SIZES:
            thru, lat_ms, lat_per_img = results[device][bs]
            f.write(f"  batch={bs}: {thru:,.0f} img/s, {lat_per_img*1000:.3f} us/img\n")
        f.write("\n")
    f.write("FPGA Theoretical:\n")
    f.write(f"  Sequential:  {HW_SEQ_THRU:,.0f} img/s, {HW_SEQ_LAT_MS*1000:.0f} us/img\n")
    f.write(f"  Parallel:    {HW_PAR_THRU:,.0f} img/s, {HW_PAR_LAT_MS*1000:.0f} us/img\n")
    f.write(f"  Pipelined:   {HW_PIPE_THRU:,.0f} img/s, {HW_PAR_LAT_MS*1000:.0f} us/img\n")

print("  Results saved to benchmark_results.txt")
print("="*70)
