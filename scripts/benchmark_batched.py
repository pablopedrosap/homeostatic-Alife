#!/usr/bin/env python3
"""Benchmark numpy vs torch batched homeostasis at various scales.

Demonstrates GPU-readiness by comparing throughput (env-steps/s) across
backends and n_envs from 256 to 65536.

Usage
-----
    python scripts/benchmark_batched.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.config import AgentConfig
from homeostatic_colony.dynamics.batched_homeostasis import HomeostasisModule

try:
    import torch
    from homeostatic_colony.dynamics.batched_homeostasis_torch import HomeostasisModuleTorch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

N_ENVS_LIST = [256, 1024, 4096, 16384, 65536]
N_STEPS = 1000
WARMUP = 50


def benchmark_numpy(n_envs: int, cfg: AgentConfig) -> float:
    """Returns env-steps/s for numpy backend."""
    module = HomeostasisModule(n_envs=n_envs, cfg=cfg)
    module.reset()

    rng = np.random.default_rng(42)
    move = rng.uniform(0, 3, size=n_envs).astype(np.float32)
    nutrients = rng.uniform(0, 0.5, size=n_envs).astype(np.float32)
    heat = rng.uniform(0, 0.1, size=n_envs).astype(np.float32)
    toxins = rng.uniform(0, 0.05, size=n_envs).astype(np.float32)
    resting = rng.random(n_envs) < 0.1

    # Warmup
    for _ in range(WARMUP):
        module.step(move, nutrients, heat, toxins, resting)

    module.reset()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        module.step(move, nutrients, heat, toxins, resting)
    elapsed = time.perf_counter() - t0
    return n_envs * N_STEPS / elapsed


def benchmark_torch(n_envs: int, cfg: AgentConfig, device: str) -> float:
    """Returns env-steps/s for torch backend on given device."""
    dev = torch.device(device)
    module = HomeostasisModuleTorch(n_envs=n_envs, cfg=cfg, device=device)
    module.reset()

    rng = np.random.default_rng(42)
    move = torch.from_numpy(rng.uniform(0, 3, size=n_envs).astype(np.float32)).to(dev)
    nutrients = torch.from_numpy(rng.uniform(0, 0.5, size=n_envs).astype(np.float32)).to(dev)
    heat = torch.from_numpy(rng.uniform(0, 0.1, size=n_envs).astype(np.float32)).to(dev)
    toxins = torch.from_numpy(rng.uniform(0, 0.05, size=n_envs).astype(np.float32)).to(dev)
    resting = torch.from_numpy(rng.random(n_envs) < 0.1).to(dev)

    # Warmup
    for _ in range(WARMUP):
        module.step(move, nutrients, heat, toxins, resting)
    if device == "cuda":
        torch.cuda.synchronize()

    module.reset()
    t0 = time.perf_counter()
    for _ in range(N_STEPS):
        module.step(move, nutrients, heat, toxins, resting)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return n_envs * N_STEPS / elapsed


def correctness_check(cfg: AgentConfig) -> None:
    """Verify numpy and torch produce identical outputs."""
    n = 128
    rng = np.random.default_rng(123)

    np_mod = HomeostasisModule(n_envs=n, cfg=cfg)
    torch_mod = HomeostasisModuleTorch(n_envs=n, cfg=cfg, device="cpu")

    move = rng.uniform(0, 3, size=n).astype(np.float32)
    nutrients = rng.uniform(0, 0.5, size=n).astype(np.float32)
    heat = rng.uniform(0, 0.1, size=n).astype(np.float32)
    toxins = rng.uniform(0, 0.05, size=n).astype(np.float32)
    resting = rng.random(n) < 0.1

    move_t = torch.from_numpy(move)
    nutrients_t = torch.from_numpy(nutrients)
    heat_t = torch.from_numpy(heat)
    toxins_t = torch.from_numpy(toxins)
    resting_t = torch.from_numpy(resting)

    for _ in range(10):
        np_mod.step(move, nutrients, heat, toxins, resting)
        torch_mod.step(move_t, nutrients_t, heat_t, toxins_t, resting_t)

    tol = 1e-5
    assert np.allclose(np_mod.energy, torch_mod.energy.numpy(), atol=tol), \
        f"Energy mismatch: max diff = {np.max(np.abs(np_mod.energy - torch_mod.energy.numpy()))}"
    assert np.allclose(np_mod.temp, torch_mod.temp.numpy(), atol=tol), \
        f"Temp mismatch: max diff = {np.max(np.abs(np_mod.temp - torch_mod.temp.numpy()))}"
    assert np.allclose(np_mod.damage, torch_mod.damage.numpy(), atol=tol), \
        f"Damage mismatch: max diff = {np.max(np.abs(np_mod.damage - torch_mod.damage.numpy()))}"

    # Check hybrid reward
    base_r = np.ones(n, dtype=np.float32)
    prev_e = np.full(n, cfg.e_init, dtype=np.float32)
    prev_t = np.full(n, cfg.t_init, dtype=np.float32)
    prev_d = np.full(n, cfg.d_init, dtype=np.float32)

    np_r = np_mod.compute_hybrid_reward(base_r, prev_e, prev_t, prev_d)
    torch_r = torch_mod.compute_hybrid_reward(
        torch.from_numpy(base_r),
        torch.from_numpy(prev_e),
        torch.from_numpy(prev_t),
        torch.from_numpy(prev_d),
    ).numpy()
    assert np.allclose(np_r, torch_r, atol=tol), \
        f"Reward mismatch: max diff = {np.max(np.abs(np_r - torch_r))}"

    print("Correctness check PASSED (numpy == torch within 1e-5)")


def main():
    cfg = AgentConfig(heat_move=0.005, cooling_rate=0.010, t_fatal=1.2)

    if HAS_TORCH:
        print("Running correctness check...")
        correctness_check(cfg)
        print()

    has_cuda = HAS_TORCH and torch.cuda.is_available()

    print(f"{'n_envs':>8}  {'NumPy CPU':>15}  ", end="")
    if HAS_TORCH:
        print(f"{'Torch CPU':>15}  ", end="")
    if has_cuda:
        print(f"{'Torch CUDA':>15}  ", end="")
    print()
    print("-" * 70)

    for n_envs in N_ENVS_LIST:
        np_rate = benchmark_numpy(n_envs, cfg)
        print(f"{n_envs:>8}  {np_rate:>12,.0f} /s  ", end="")

        if HAS_TORCH:
            torch_rate = benchmark_torch(n_envs, cfg, "cpu")
            print(f"{torch_rate:>12,.0f} /s  ", end="")

        if has_cuda:
            cuda_rate = benchmark_torch(n_envs, cfg, "cuda")
            print(f"{cuda_rate:>12,.0f} /s  ", end="")

        print()

    print()
    print("Throughput = n_envs * 1000 steps / wall_time")


if __name__ == "__main__":
    main()
