"""
Predictive world model for anticipatory control.

This module provides:
1) Legacy MLP dynamics predictor (backward-compatible).
2) Latent world-model predictor (encoder + transition + predictive heads).
3) Trainer utilities for online fitting + simple predictive planning.

The world-model path enables moving from reactive control to predictive control:
actions can be evaluated by imagined consequences before execution.
"""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)


class MLPDynamicsPredictor(nn.Module):
    """Legacy MLP predictor for next internal state [E, T, D]."""

    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 64):
        super().__init__()
        input_dim = 3 + obs_dim + action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class WorldModelPredictor(nn.Module):
    """
    Latent world model: state encoder + transition + prediction heads.

    Input:
      concat(internal[3], external[obs_dim], action[action_dim])
    Pipeline:
      state -> latent z -> transition(z, action) -> z_next -> heads
    Output:
      mean prediction of next internal [E, T, D]
      optional predictive log-variance for uncertainty-aware control
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden: int = 128,
        predict_logvar: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.predict_logvar = predict_logvar

        state_dim = 3 + obs_dim
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
            nn.ReLU(),
        )
        self.transition = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )
        self.mean_head = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3),
        )
        self.logvar_head = None
        if predict_logvar:
            self.logvar_head = nn.Sequential(
                nn.Linear(latent_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 3),
            )

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        i_end = 3
        e_end = 3 + self.obs_dim
        internal = x[:, :i_end]
        external = x[:, i_end:e_end]
        action = x[:, e_end:e_end + self.action_dim]
        return internal, external, action

    def encode_state(self, internal: torch.Tensor, external: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([internal, external], dim=1))

    def transition_latent(self, z: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        delta = self.transition(torch.cat([z, action], dim=1))
        # Residual transition for stable short-horizon prediction.
        return z + delta

    def predict_with_uncertainty(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        internal, external, action = self._split(x)
        z = self.encode_state(internal, external)
        z_next = self.transition_latent(z, action)
        mean = self.mean_head(z_next)

        if self.logvar_head is None:
            logvar = torch.zeros_like(mean)
        else:
            logvar = self.logvar_head(z_next).clamp(min=-8.0, max=4.0)
        return mean, logvar

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean, _ = self.predict_with_uncertainty(x)
        return mean


# Backward-compatible name used by existing scripts.
class DynamicsPredictor(MLPDynamicsPredictor):
    pass


class PredictorTrainer:
    """Trainer + predictive-control helpers for dynamics models."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 64,
        lr: float = 1e-3,
        buffer_size: int = 50_000,
        batch_size: int = 128,
        device: str = "cpu",
        architecture: Literal["mlp", "world_model"] = "world_model",
        latent_dim: int = 64,
        predict_uncertainty: bool = True,
    ):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden = hidden
        self.lr = lr
        self.architecture = architecture
        self.latent_dim = latent_dim
        self.predict_uncertainty = bool(predict_uncertainty)

        self.device = torch.device(device)
        self.model = self._build_model().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.buffer: deque[tuple] = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.loss_history: list[float] = []
        self._error_ema = 0.0

    def _build_model(self) -> nn.Module:
        if self.architecture == "mlp":
            return MLPDynamicsPredictor(self.obs_dim, self.action_dim, self.hidden)
        if self.architecture == "world_model":
            return WorldModelPredictor(
                self.obs_dim,
                self.action_dim,
                latent_dim=self.latent_dim,
                hidden=max(self.hidden, 64),
                predict_logvar=self.predict_uncertainty,
            )
        raise ValueError(f"Unknown predictor architecture: {self.architecture}")

    def add_transition(
        self,
        internal: np.ndarray,
        external: np.ndarray,
        action: np.ndarray,
        next_internal: np.ndarray,
    ) -> None:
        self.buffer.append((
            np.asarray(internal, dtype=np.float32).copy(),
            np.asarray(external, dtype=np.float32).copy(),
            np.asarray(action, dtype=np.float32).copy(),
            np.asarray(next_internal, dtype=np.float32).copy(),
        ))

    def _sample_batch(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
        if len(self.buffer) < self.batch_size:
            return None

        idx = np.random.choice(len(self.buffer), self.batch_size, replace=False)
        batch = [self.buffer[i] for i in idx]
        internals = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32, device=self.device)
        externals = torch.tensor(np.array([b[1] for b in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array([b[2] for b in batch]), dtype=torch.float32, device=self.device)
        targets = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32, device=self.device)
        return internals, externals, actions, targets

    def train_step(self) -> float | None:
        sampled = self._sample_batch()
        if sampled is None:
            return None
        internals, externals, actions, targets = sampled

        x = torch.cat([internals, externals, actions], dim=1)

        if self.architecture == "world_model" and self.predict_uncertainty:
            mean, logvar = self.model.predict_with_uncertainty(x)  # type: ignore[attr-defined]
            inv_var = torch.exp(-logvar)
            # Diagonal Gaussian NLL + small MSE stabilizer.
            nll = 0.5 * ((targets - mean) ** 2 * inv_var + logvar)
            loss = nll.mean() + 0.1 * self.loss_fn(mean, targets)
        else:
            pred = self.model(x)
            loss = self.loss_fn(pred, targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        lv = float(loss.item())
        self.loss_history.append(lv)
        return lv

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
        self.model.eval()
        model_uncertainty = 0.0
        with torch.no_grad():
            x = torch.tensor(
                np.concatenate([internal, external, action]),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

            if self.architecture == "world_model" and self.predict_uncertainty:
                pred_t, logvar_t = self.model.predict_with_uncertainty(x)  # type: ignore[attr-defined]
                pred = pred_t.squeeze(0).cpu().numpy()
                var_t = torch.exp(logvar_t).mean()
                model_uncertainty = float(torch.sqrt(var_t).cpu().item())
            else:
                pred = self.model(x).squeeze(0).cpu().numpy()

        pred_dev = (
            w_energy * abs(e_target - float(pred[0]))
            + w_temp * abs(float(pred[1]) - t_target)
            + w_damage * float(pred[2])
        )

        uncertainty = 0.5 * self._error_ema + model_uncertainty
        risk_score = float(pred_dev + uncertainty)
        return risk_score, pred

    def imagine_rollout(
        self,
        internal: np.ndarray,
        external: np.ndarray,
        action_sequence: np.ndarray,
        gamma: float = 0.95,
        e_target: float = 0.7,
        t_target: float = 0.3,
        w_energy: float = 1.0,
        w_temp: float = 1.0,
        w_damage: float = 1.5,
    ) -> dict:
        """
        Roll forward imagined consequences of an action sequence.

        External observation is treated as locally stationary over short horizons.
        """
        seq = np.asarray(action_sequence, dtype=np.float32)
        curr_internal = np.asarray(internal, dtype=np.float32).copy()
        ext = np.asarray(external, dtype=np.float32)

        preds: list[np.ndarray] = []
        risks: list[float] = []
        total = 0.0
        discount = 1.0
        for action in seq:
            risk, next_internal = self.predict_risk(
                curr_internal,
                ext,
                action,
                e_target=e_target,
                t_target=t_target,
                w_energy=w_energy,
                w_temp=w_temp,
                w_damage=w_damage,
            )
            preds.append(next_internal.copy())
            risks.append(risk)
            total += discount * risk
            discount *= gamma
            curr_internal = next_internal

        return {
            "predicted_internals": np.array(preds, dtype=np.float32),
            "step_risks": np.array(risks, dtype=np.float32),
            "discounted_total_risk": float(total),
        }

    def plan_action(
        self,
        internal: np.ndarray,
        external: np.ndarray,
        action_candidates: np.ndarray,
        horizon: int = 3,
        n_rollouts: int = 8,
        noise_std: float = 0.2,
        gamma: float = 0.95,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Select action via random-shooting MPC over imagined trajectories.

        Returns:
            best_action, best_score, candidate_scores
        """
        candidates = np.asarray(action_candidates, dtype=np.float32)
        if candidates.ndim == 1:
            candidates = candidates[None, :]
        if candidates.shape[0] == 0:
            raise ValueError("action_candidates cannot be empty")

        horizon = max(1, int(horizon))
        n_rollouts = max(1, int(n_rollouts))
        scores = np.zeros(candidates.shape[0], dtype=np.float32)

        for i, a0 in enumerate(candidates):
            rollout_scores = []
            for _ in range(n_rollouts):
                seq = [a0]
                for _h in range(1, horizon):
                    seq.append(np.clip(a0 + np.random.normal(0.0, noise_std, size=a0.shape), -1.0, 1.0))
                imagined = self.imagine_rollout(
                    internal,
                    external,
                    np.array(seq, dtype=np.float32),
                    gamma=gamma,
                )
                rollout_scores.append(float(imagined["discounted_total_risk"]))
            scores[i] = float(np.mean(rollout_scores))

        best_idx = int(np.argmin(scores))
        return candidates[best_idx].copy(), float(scores[best_idx]), scores

    def update_error_ema(self, actual: np.ndarray, predicted: np.ndarray, alpha: float = 0.05) -> None:
        error = float(np.mean((np.asarray(actual) - np.asarray(predicted)) ** 2))
        self._error_ema = alpha * error + (1 - alpha) * self._error_ema

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
                "error_ema": self._error_ema,
                "meta": {
                    "architecture": self.architecture,
                    "obs_dim": self.obs_dim,
                    "action_dim": self.action_dim,
                    "hidden": self.hidden,
                    "lr": self.lr,
                    "latent_dim": self.latent_dim,
                    "predict_uncertainty": self.predict_uncertainty,
                    "batch_size": self.batch_size,
                },
            },
            path,
        )
        logger.info(f"Predictor saved to {path}")

    def load(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device, weights_only=True)
        meta = checkpoint.get("meta", {})
        if meta:
            new_arch = meta.get("architecture", self.architecture)
            new_obs = int(meta.get("obs_dim", self.obs_dim))
            new_act = int(meta.get("action_dim", self.action_dim))
            new_hidden = int(meta.get("hidden", self.hidden))
            new_latent = int(meta.get("latent_dim", self.latent_dim))
            new_unc = bool(meta.get("predict_uncertainty", self.predict_uncertainty))

            needs_rebuild = (
                new_arch != self.architecture
                or new_obs != self.obs_dim
                or new_act != self.action_dim
                or new_hidden != self.hidden
                or new_latent != self.latent_dim
                or new_unc != self.predict_uncertainty
            )
            if needs_rebuild:
                self.architecture = new_arch
                self.obs_dim = new_obs
                self.action_dim = new_act
                self.hidden = new_hidden
                self.latent_dim = new_latent
                self.predict_uncertainty = new_unc
                self.model = self._build_model().to(self.device)
                self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        self.model.load_state_dict(checkpoint["model_state"])
        opt_state = checkpoint.get("optimizer_state")
        if opt_state:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception:
                logger.warning("Could not load optimizer state; continuing with fresh optimizer.")
        self.loss_history = checkpoint.get("loss_history", [])
        self._error_ema = checkpoint.get("error_ema", 0.0)
        logger.info(f"Predictor loaded from {path}")
