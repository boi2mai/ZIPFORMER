# Design Space Exploration and Memory Architecture Optimization for Zipformer Accelerator.

An automated Design Space Exploration (DSE) and on-chip memory tiling optimization framework tailored for edge Zipformer hardware accelerators.

---

## Overview
Zipformer introduces deep asymmetric U-Net structures where downsampled sequence lengths ($T \ll D$) cause massive parameter explosions in deep stacks. This project provides an automated, end-to-end framework combining an analytical DSE sweeper, a cycle/transaction-accurate behavioral profiler, and microarchitectural optimizations (FFW, Nonlinear Attention, and Self-Attention) to constrain peak SRAM usage strictly within **512 KB** on a 256-PE baseline without compute starvation.

## Project Structure
```text
├── configs/
│   └── golden_configs.json          # Best strategies, tile shapes, and layer configurations
├── modules/
│   ├── ffw.py                       # FeedForward module modeling & residual accumulation
│   ├── nonlin.py                    # Nonlinear attention with fused branches
│   └── self_attention.py            # Multi-head self-attention engine
├── profiler/
│   ├── cost_model.py                # Analytical mathematical SRAM and MAC estimation
│   └── behavioral_profiler.py       # Nested-loop hardware simulation & transaction counters
├── reports/
│   └── dse_results_history.xlsx     # Sweep logs, memory traffic, and operational intensity metrics
├── tests/
│   ├── verify_all_cases.py          # 18-case cross-verification script against PyTorch Float64
│   └── golden_reference.py          # PyTorch baseline model wrapper
├── sweep_tiling.py                  # Main CLI entrypoint for automated DSE exploration
├── requirements.txt
└── README.md
```

## Key Highlights & Hardware Optimizations
* **Channel-First Input-Stationary Dataflow:** Retains input activation tensors resident on-chip while streaming sliced weight matrices to exploit $T \ll D$ asymmetry.
* **Latency Hiding via Ping-Pong Buffering:** Overlaps DMA DRAM transactions with PE computation across dual memory banks.
* **Operator & Memory Fusion:** Fuses multiplexed nonlinear attention branches to eliminate intermediate tensor allocations ($Z$), cutting peak SRAM by over 50%.
* **Two-Stage Automated DSE Framework:**
  * **Analytical Sweeper:** Rapidly prunes the parameter space ($tile\_T, tile\_H, tile\_C$) using mathematical models against the 512 KB SRAM threshold.
  * **Behavioral Profiler:** Simulates hardware loop execution to log exact transaction counts (`g_loads`, `g_stores`, `l_loads`, `l_stores`, and MACs).
* **Bit-Exact Architectural Verification:** Cross-verified against Dan Povey's official PyTorch Zipformer implementation under Float64 evaluation mode across all 6 stacks $\times$ 3 modules (18/18 cases PASSED with maximum absolute error $< 10^{-10}$).

## Outputs & Artifacts
* `golden_configs.json`: Contains the optimal tiling dimensions and hardware deployment parameters.
* `dse_results_history.xlsx`: Detailed profiling history logging execution timestamps, DRAM access counts, and operational intensity ($\text{MACs/Load}$).

## Contributors
* **Huy Le** ([@boi2mai](https://github.com/boi2mai))
