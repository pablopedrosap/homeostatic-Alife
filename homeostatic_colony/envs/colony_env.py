"""
Colony simulation: multiple simple cells with local coupling.

This is a non-RL particle simulation where many cells (100-300) interact
through local nutrient competition, toxin exposure, and a diffusing
communication signal (quorum-sensing-like).

Each cell follows simple biased-random-walk rules, NOT a learned policy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from ..fields import FieldManager

logger = logging.getLogger(__name__)


@dataclass
class CellState:
    """State of a single cell in the colony."""
    pos: np.ndarray
    energy: float
    temp: float
    damage: float
    alive: bool = True
    age: int = 0


@dataclass
class ColonyConfig:
    """Configuration for colony simulation."""
    width: float = 30.0
    height: float = 30.0
    n_initial_cells: int = 100
    max_cells: int = 500
    n_nutrients: int = 8
    n_toxins: int = 5
    n_cool: int = 3
    nutrient_radius: float = 3.0
    toxin_radius: float = 3.5
    cool_radius: float = 2.5

    # Cell dynamics (simplified)
    basal_cost: float = 0.002
    move_cost: float = 0.005
    nutrient_gain: float = 0.1
    heat_move: float = 0.005
    heat_hazard: float = 0.04
    cooling_rate: float = 0.012
    dmg_toxin: float = 0.03
    dmg_overheat: float = 0.04
    repair_rate: float = 0.008
    t_crit: float = 0.7
    t_fatal: float = 1.2
    d_max: float = 1.0

    # Division / death
    division_energy: float = 0.85
    division_cost: float = 0.4

    # Signal
    signal_diffusion_rate: float = 0.1
    signal_decay: float = 0.95
    signal_threshold: float = 0.3  # cells respond when local signal > threshold

    # Simulation
    max_steps: int = 3000
    seed: int = 42
    use_signaling: bool = True


class ColonySimulation:
    """
    Simple colony simulation with local coupling and optional signaling.

    Cells move via biased random walk (toward nutrients, away from toxins).
    When signaling is enabled, cells secrete a chemical when stressed
    (high damage/temp) and neighbors bias movement away from signal sources.
    """

    def __init__(self, config: ColonyConfig | None = None):
        self.cfg = config or ColonyConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.cells: list[CellState] = []
        self.signal_field: np.ndarray | None = None  # 2D grid
        self.fields: FieldManager | None = None
        self.step_count = 0
        self._signal_res = 60  # grid resolution for signal field

        # History for analysis
        self.history: list[dict] = []

    def reset(self) -> None:
        """Initialize the colony."""
        self.rng = np.random.default_rng(self.cfg.seed)
        self.step_count = 0
        self.history = []

        # Create fields
        self.fields = FieldManager(self.cfg.width, self.cfg.height, self.rng)
        self.fields.generate(
            n_nutrients=self.cfg.n_nutrients,
            n_toxins=self.cfg.n_toxins,
            n_cool=self.cfg.n_cool,
            nutrient_radius=self.cfg.nutrient_radius,
            toxin_radius=self.cfg.toxin_radius,
            cool_radius=self.cfg.cool_radius,
            nutrient_strength=1.0,
            toxin_strength=1.0,
            cool_strength=1.0,
        )

        # Spawn cells
        self.cells = []
        for _ in range(self.cfg.n_initial_cells):
            pos = np.array([
                self.rng.uniform(1, self.cfg.width - 1),
                self.rng.uniform(1, self.cfg.height - 1),
            ], dtype=np.float32)
            self.cells.append(CellState(
                pos=pos,
                energy=self.rng.uniform(0.5, 0.8),
                temp=self.rng.uniform(0.2, 0.4),
                damage=0.0,
            ))

        # Signal field
        self.signal_field = np.zeros((self._signal_res, self._signal_res), dtype=np.float32)

    def step(self) -> dict:
        """Advance one time step."""
        assert self.fields is not None
        self.step_count += 1
        cfg = self.cfg

        # Diffuse and decay signal field
        if cfg.use_signaling and self.signal_field is not None:
            self.signal_field *= cfg.signal_decay
            # Simple diffusion via convolution
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            from scipy.ndimage import convolve
            diffused = convolve(self.signal_field, kernel * cfg.signal_diffusion_rate, mode="constant")
            self.signal_field = np.clip(self.signal_field + diffused, 0.0, 10.0)

        new_cells = []
        for cell in self.cells:
            if not cell.alive:
                continue

            # Sample local fields
            fv = self.fields.sample_at(cell.pos)
            nut_grad = self.fields.nutrient_gradient(cell.pos)
            tox_grad = self.fields.toxin_gradient(cell.pos)

            # Biased random walk
            bias = nut_grad * 0.5 - tox_grad * 0.8

            # Signal avoidance (move away from high signal)
            if cfg.use_signaling and self.signal_field is not None:
                sig_grad = self._signal_gradient(cell.pos)
                local_sig = self._sample_signal(cell.pos)
                if local_sig > cfg.signal_threshold:
                    bias -= sig_grad * 1.0

            noise = self.rng.normal(0, 0.15, size=2).astype(np.float32)
            move = bias + noise
            move_mag = float(np.linalg.norm(move))
            if move_mag > 0.3:
                move = move / move_mag * 0.3  # cap speed
                move_mag = 0.3

            cell.pos = np.clip(
                cell.pos + move,
                [0, 0],
                [cfg.width, cfg.height],
            )
            cell.age += 1

            # Update internal state
            cell.energy = np.clip(
                cell.energy - cfg.basal_cost - cfg.move_cost * move_mag + cfg.nutrient_gain * fv["nutrient"],
                0.0, 1.0,
            )
            cell.temp = np.clip(
                cell.temp + cfg.heat_move * move_mag ** 2 + cfg.heat_hazard * fv["toxin"] - cfg.cooling_rate,
                0.0, 1.5,
            )
            overheat = max(0.0, cell.temp - cfg.t_crit)
            cell.damage = np.clip(
                cell.damage + cfg.dmg_toxin * fv["toxin"] + cfg.dmg_overheat * overheat - cfg.repair_rate,
                0.0, cfg.d_max,
            )

            # Secrete signal when stressed
            if cfg.use_signaling and self.signal_field is not None:
                stress = cell.damage + max(0, cell.temp - cfg.t_crit)
                if stress > 0.2:
                    self._deposit_signal(cell.pos, stress * 0.5)

            # Check death
            if cell.energy <= 0 or cell.damage >= cfg.d_max or cell.temp >= cfg.t_fatal:
                cell.alive = False
                continue

            # Division
            if cell.energy > cfg.division_energy and len(self.cells) + len(new_cells) < cfg.max_cells:
                cell.energy -= cfg.division_cost
                offset = self.rng.normal(0, 0.5, size=2).astype(np.float32)
                child_pos = np.clip(cell.pos + offset, [0, 0], [cfg.width, cfg.height])
                new_cells.append(CellState(
                    pos=child_pos,
                    energy=cell.energy * 0.5,
                    temp=cell.temp,
                    damage=0.0,
                ))
                cell.energy *= 0.5

        self.cells.extend(new_cells)
        alive_cells = [c for c in self.cells if c.alive]

        # Record history
        record = {
            "step": self.step_count,
            "population": len(alive_cells),
            "mean_energy": np.mean([c.energy for c in alive_cells]) if alive_cells else 0,
            "mean_damage": np.mean([c.damage for c in alive_cells]) if alive_cells else 0,
            "mean_temp": np.mean([c.temp for c in alive_cells]) if alive_cells else 0,
            "births": len(new_cells),
            "deaths": sum(1 for c in self.cells if not c.alive) - sum(1 for r in self.history if "deaths" in r),
        }
        self.history.append(record)
        return record

    def _sample_signal(self, pos: np.ndarray) -> float:
        if self.signal_field is None:
            return 0.0
        gx = int(pos[0] / self.cfg.width * (self._signal_res - 1))
        gy = int(pos[1] / self.cfg.height * (self._signal_res - 1))
        gx = np.clip(gx, 0, self._signal_res - 1)
        gy = np.clip(gy, 0, self._signal_res - 1)
        return float(self.signal_field[gy, gx])

    def _deposit_signal(self, pos: np.ndarray, amount: float) -> None:
        if self.signal_field is None:
            return
        gx = int(pos[0] / self.cfg.width * (self._signal_res - 1))
        gy = int(pos[1] / self.cfg.height * (self._signal_res - 1))
        gx = np.clip(gx, 0, self._signal_res - 1)
        gy = np.clip(gy, 0, self._signal_res - 1)
        self.signal_field[gy, gx] += amount

    def _signal_gradient(self, pos: np.ndarray, eps: float = 0.5) -> np.ndarray:
        grad = np.zeros(2, dtype=np.float32)
        for dim in range(2):
            pp = pos.copy()
            pm = pos.copy()
            pp[dim] += eps
            pm[dim] -= eps
            grad[dim] = (self._sample_signal(pp) - self._sample_signal(pm)) / (2 * eps)
        return grad

    def run(self, max_steps: int | None = None) -> list[dict]:
        """Run the full simulation."""
        self.reset()
        steps = max_steps or self.cfg.max_steps
        for _ in range(steps):
            record = self.step()
            alive = record["population"]
            if alive == 0:
                logger.info(f"Colony extinct at step {self.step_count}")
                break
        return self.history
