"""
Predictive dynamics model for anticipatory risk estimation.

A small MLP that predicts next internal state [E, T, D] given
current observation and action. This enables "predictive self-maintenance"
— the agent can estimate consequences of actions before taking them.

This is functional risk anticipation, NOT a claim of subjective fear.
"""

from __future__ import annotations

import logging
from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class DynamicsPredictor(nn.Module):
    """
    MLP that predicts next internal state from current obs + action.

    Input:  concat(internal_state[3], external_obs[n], action[m])
    Output: predicted next internal state [E, T, D] (3 dims)
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        input_dim = 3 + obs_dim + action_dim  # 3 for internal [E,T,D]
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),  # predict [E, T, D]
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictorTrainer:
    """Online trainer for the dynamics predictor with replay buffer."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 64,
        lr: float = 1e-3,
        buffer_size: int = 50_000,
        batch_size: int = 128,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = DynamicsPredictor(obs_dim, action_dim, hidden).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer: deque[tuple] = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.loss_history: list[float] = []
        self._error_ema = 0.0  # exponential moving average of prediction error

    def add_transition(
        self,
        internal: np.ndarray,
        external: np.ndarray,
        action: np.ndarray,
        next_internal: np.ndarray,
    ) -> None:
        """Store a transition in the replay buffer."""
        self.buffer.append((
            internal.copy(),
            external.copy(),
            action.copy(),
            next_internal.copy(),
        ))

    def train_step(self) -> float | None:
        """Sample a batch and do one gradient step. Returns loss or None."""
        if len(self.buffer) < self.batch_size:
            return None

        indices = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]

        internals = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        externals = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32, device=self.device)
        targets = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device)

        x = torch.cat([internals, externals, actions], dim=1)
        pred = self.model(x)
        loss = self.loss_fn(pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        loss_val = loss.item()
        self.loss_history.append(loss_val)
        return loss_val

    def predict_risk(
        self,
        internal: np.ndarray,
        external: np.ndarray,
        action: np.ndarray,
        e_target: float = 0.7,
        t_target: float = 0.3,
        w_energy: float = 1.0,
        w_temp: float = 1.0,
        w_damage: float = 1.5,
    ) -> tuple[float, np.ndarray]:
        """
        Predict next internal state and compute risk score.

        Returns:
            risk_score: predicted homeostatic deviation + uncertainty
            predicted_internal: [E, T, D] prediction
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.tensor(
                np.concatenate([internal, external, action]),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            pred = self.model(x).squeeze(0).cpu().numpy()

        # Predicted homeostatic deviation
        pred_dev = (
            w_energy * abs(e_target - pred[0])
            + w_temp * abs(pred[1] - t_target)
            + w_damage * pred[2]
        )

        # Uncertainty proxy: recent prediction error EMA
        uncertainty = self._error_ema * 0.5
        risk_score = float(pred_dev + uncertainty)

        return risk_score, pred

    def update_error_ema(self, actual: np.ndarray, predicted: np.ndarray, alpha: float = 0.05) -> None:
        """Update the exponential moving average of prediction error."""
        error = float(np.mean((actual - predicted) ** 2))
        self._error_ema = alpha * error + (1 - alpha) * self._error_ema

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss_history": self.loss_history,
            "error_ema": self._error_ema,
        }, path)
        logger.info(f"Predictor saved to {path}")

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.loss_history = checkpoint.get("loss_history", [])
        self._error_ema = checkpoint.get("error_ema", 0.0)
        logger.info(f"Predictor loaded from {path}")
