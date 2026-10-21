# Proposed HANDOFF.md / STATUS.md entry (not applied; the owner applies ledger edits)

**2026-10-12 — GPU port (`studies/gpu_port_20261012`).** First study on the GPU machine (RTX 3070 8 GB, Ryzen 7
5800X), with environment `.venv` (torch 2.14.1+cu130) and MNIST + CIFAR-10 in `datasets/`.
- The device-ready PARSEC-L copy (`gpu_parsec`, random numbers from CPU generators) equals the original bitwise on
  the CPU.
- GPU runs repeat bitwise. One GPU float64 step agrees with the CPU to 3.3e-10.
- Trajectories are chaotic: device, precision and machine change single step-500 EMA losses by 0.002–0.008.
- Top-free c = 64 + EMA beats SOAP + EMA in seeds 4–6 on this machine in float64 (0.123 vs 0.133) and in GPU
  float32 (0.123 vs 0.136). Float32 is adequate by the pre-declared rule.
- Cost: top-free is about 3× SOAP per step on a 0.2M-parameter CIFAR ViT. Its probe path runs out of memory beyond
  about 2,000 probe rows × 0.2M parameters.
- Deviations D1 (replay print bug) and D2 (two cost runs) are recorded.
- Run `~/skipsoap-study-data/gpu_port_20261012/runs/20261012T084322-39ae45d3`, report
  `studies/gpu_port_20261012/RESULTS.md`.
