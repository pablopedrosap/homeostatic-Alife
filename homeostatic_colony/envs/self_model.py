"""A learned self-model in the control loop — the functional form of a "strange loop".

The robot learns, from its own experience only, to predict its own motor temperatures
``horizon`` steps ahead, and the policy observes those predictions. The loop is
self-referential: the policy's behaviour generates the data the self-model learns from, and the
self-model's predictions shape the policy's behaviour (Hofstadter, *Gödel, Escher, Bach*;
Graziano's attention-schema idea for a body). Nothing about the thermal physics is given to it.

This is NOT a claim of consciousness. It is a mechanism with a causal test: at evaluation
time the loop can be cut (``ablate=True`` → the "prediction" is just the current temperature)
to measure whether the agent actually uses its model of itself.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn


class ThermalSelfModel:
    """Online-trained forward model: body state now → own motor temperatures in ``horizon`` steps."""

    def __init__(self, obs_dim: int = 58, n_motors: int = 12, horizon: int = 100,
                 hidden: int = 64, buffer_size: int = 100_000, train_every: int = 2048,
                 grad_steps: int = 32, batch: int = 256, lr: float = 1e-3, seed: int = 0):
        torch.manual_seed(seed)
        self.horizon = horizon
        self.n = n_motors
        self.net = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, n_motors))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.X = np.zeros((buffer_size, obs_dim), np.float32)
        self.Y = np.zeros((buffer_size, n_motors), np.float32)
        self.size = self.ptr = 0
        self.pending: dict[int, deque] = {}
        self.train_every, self.grad_steps, self.batch = train_every, grad_steps, batch
        self.calls = 0
        self.frozen = False
        self.ablate = False
        self.last_loss = float("nan")
        self.rng = np.random.default_rng(seed)

    # -- inference ------------------------------------------------------------------
    def predict(self, obs: np.ndarray, temps: np.ndarray) -> np.ndarray:
        """Predicted temperatures ``horizon`` steps ahead (or current temps if the loop is cut)."""
        if self.ablate:
            return temps.astype(np.float32)
        with torch.no_grad():
            delta = self.net(torch.as_tensor(obs[None], dtype=torch.float32))[0].numpy()
        return np.clip(temps + delta, 0.0, 1.2).astype(np.float32)

    # -- learning from own experience ------------------------------------------------
    def reset_stream(self, env_id: int) -> None:
        self.pending[env_id] = deque()

    def observe(self, env_id: int, obs: np.ndarray, temps: np.ndarray) -> None:
        """Feed one step of the robot's own experience; pairs obs(t) with temps(t + horizon)."""
        if self.frozen:
            return
        q = self.pending.setdefault(env_id, deque())
        q.append((obs.astype(np.float32), temps.astype(np.float32)))
        if len(q) > self.horizon:
            o, t0 = q.popleft()
            self.X[self.ptr] = o
            self.Y[self.ptr] = temps - t0          # learn the change: easier and calibrated
            self.ptr = (self.ptr + 1) % len(self.X)
            self.size = min(self.size + 1, len(self.X))
        self.calls += 1
        if self.calls % self.train_every == 0 and self.size >= self.batch:
            self._train()

    def _train(self) -> None:
        for _ in range(self.grad_steps):
            idx = self.rng.integers(0, self.size, self.batch)
            x = torch.as_tensor(self.X[idx])
            y = torch.as_tensor(self.Y[idx])
            loss = ((self.net(x) - y) ** 2).mean()
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
        self.last_loss = float(loss.item())

    # -- persistence ----------------------------------------------------------------
    def save(self, path: Path) -> None:
        torch.save(self.net.state_dict(), path)

    def load(self, path: Path) -> "ThermalSelfModel":
        self.net.load_state_dict(torch.load(path, weights_only=True))
        self.frozen = True
        return self
