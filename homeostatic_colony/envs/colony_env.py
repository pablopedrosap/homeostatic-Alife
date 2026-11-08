"""Eco-evolutionary ALife colony simulation.

This module intentionally targets open-ended ALife dynamics (not episodic RL):
- local homeostatic control per cell (energy/temperature/damage)
- renewable but depletable resources (nutrient ecology)
- heredity + mutation + lineage tracking
- optional predictive steering (look-ahead viability control)
- optional quorum-like signaling under stress

The design goal is to surface emergent macro-dynamics (population waves,
trait diversification, lineage turnover), not just trajectory videos.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import convolve

from ..fields import FieldManager

logger = logging.getLogger(__name__)

# Trait indices (fixed order)
T_FORAGE = 0
T_TOXIN_AVOID = 1
T_SPEED = 2
T_COOLING = 3
T_DIVISION = 4
T_SIGNAL_EMIT = 5
T_SIGNAL_AVOID = 6

TRAIT_NAMES = (
    "forage",
    "toxin_avoid",
    "speed",
    "cooling",
    "division",
    "signal_emit",
    "signal_avoid",
)
N_TRAITS = len(TRAIT_NAMES)
TRAIT_BASE = np.array([1.0, 1.0, 1.0, 1.0, 0.82, 1.0, 1.0], dtype=np.float32)
TRAIT_MIN = np.array([0.4, 0.4, 0.4, 0.6, 0.60, 0.2, 0.2], dtype=np.float32)
TRAIT_MAX = np.array([2.0, 2.2, 1.8, 1.6, 0.95, 2.5, 2.5], dtype=np.float32)


@dataclass
class CellState:
    """State of a single cell in the colony."""
    cell_id: int
    lineage_id: int
    parent_id: int | None
    pos: np.ndarray
    energy: float
    temp: float
    damage: float
    traits: np.ndarray = field(default_factory=lambda: TRAIT_BASE.copy())
    alive: bool = True
    age: int = 0
    generation: int = 0
    stress_memory: float = 0.0
    death_cause: str = ""


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

    # Eco-dynamics
    enable_resource_dynamics: bool = True
    nutrient_regrowth: float = 0.02
    nutrient_consumption: float = 0.035
    nutrient_floor: float = 0.0
    cool_zone_cooling: float = 0.02
    repair_min_energy: float = 0.25

    # Evolution
    enable_evolution: bool = True
    initial_trait_std: float = 0.08
    mutation_std: float = 0.06
    min_division_age: int = 20
    division_jitter: float = 0.03

    # Predictive steering (micro world-model)
    predictive_steering: bool = True
    lookahead_steps: int = 2
    risk_temp_weight: float = 1.0
    risk_damage_weight: float = 1.2
    risk_energy_weight: float = 0.8


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
        self.nutrient_stock: np.ndarray | None = None
        self.fields: FieldManager | None = None
        self.step_count = 0
        self._signal_res = 60  # grid resolution for signal field
        self._next_cell_id = 0

        # History for analysis
        self.history: list[dict] = []
        self._deaths_total = 0
        self._deaths_by_cause = {
            "starvation": 0,
            "overheating": 0,
            "damage_overload": 0,
        }
        self._births_total = 0

    def reset(self) -> None:
        """Initialize the colony."""
        self.rng = np.random.default_rng(self.cfg.seed)
        self.step_count = 0
        self.history = []
        self._next_cell_id = 0
        self._deaths_total = 0
        self._births_total = 0
        self._deaths_by_cause = {
            "starvation": 0,
            "overheating": 0,
            "damage_overload": 0,
        }

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
            cell_id = self._new_cell_id()
            self.cells.append(CellState(
                cell_id=cell_id,
                lineage_id=cell_id,
                parent_id=None,
                pos=pos,
                energy=self.rng.uniform(0.5, 0.8),
                temp=self.rng.uniform(0.2, 0.4),
                damage=0.0,
                traits=self._init_traits(),
            ))

        # Signal field
        self.signal_field = np.zeros((self._signal_res, self._signal_res), dtype=np.float32)
        # Nutrient resources (per source stock)
        self.nutrient_stock = np.ones(len(self.fields.nutrients), dtype=np.float32)

    def _new_cell_id(self) -> int:
        cid = self._next_cell_id
        self._next_cell_id += 1
        return cid

    def _init_traits(self) -> np.ndarray:
        traits = TRAIT_BASE + self.rng.normal(0.0, self.cfg.initial_trait_std, size=N_TRAITS).astype(np.float32)
        return np.clip(traits, TRAIT_MIN, TRAIT_MAX)

    def _mutate_traits(self, parent_traits: np.ndarray) -> np.ndarray:
        if not self.cfg.enable_evolution:
            return parent_traits.copy()
        child = parent_traits + self.rng.normal(0.0, self.cfg.mutation_std, size=N_TRAITS).astype(np.float32)
        return np.clip(child, TRAIT_MIN, TRAIT_MAX)

    def _nutrient_intensity(self, pos: np.ndarray) -> float:
        assert self.fields is not None
        if self.nutrient_stock is None:
            return float(sum(s.intensity_at(pos) for s in self.fields.nutrients))
        val = 0.0
        for i, src in enumerate(self.fields.nutrients):
            val += float(self.nutrient_stock[i]) * src.intensity_at(pos)
        return float(val)

    def _nutrient_gradient(self, pos: np.ndarray, eps: float = 0.3) -> np.ndarray:
        grad = np.zeros(2, dtype=np.float32)
        for dim in range(2):
            pp = pos.copy()
            pm = pos.copy()
            pp[dim] += eps
            pm[dim] -= eps
            grad[dim] = (self._nutrient_intensity(pp) - self._nutrient_intensity(pm)) / (2 * eps)
        return grad

    def _sample_fields(self, pos: np.ndarray) -> dict[str, float]:
        assert self.fields is not None
        vals = self.fields.sample_at(pos)
        vals["nutrient"] = self._nutrient_intensity(pos)
        return vals

    def _regrow_resources(self) -> None:
        if not self.cfg.enable_resource_dynamics or self.nutrient_stock is None:
            return
        self.nutrient_stock += self.cfg.nutrient_regrowth * (1.0 - self.nutrient_stock)
        self.nutrient_stock = np.clip(self.nutrient_stock, self.cfg.nutrient_floor, 1.0)

    def _consume_nutrients(self, pos: np.ndarray, amount: float) -> None:
        if not self.cfg.enable_resource_dynamics or self.nutrient_stock is None:
            return
        if amount <= 0:
            return
        assert self.fields is not None
        local = np.array([src.intensity_at(pos) for src in self.fields.nutrients], dtype=np.float32)
        weighted = local * self.nutrient_stock
        total = float(np.sum(weighted))
        if total <= 1e-8:
            return
        share = weighted / total
        self.nutrient_stock -= float(amount) * share
        self.nutrient_stock = np.clip(self.nutrient_stock, self.cfg.nutrient_floor, 1.0)

    def _cap_speed(self, move: np.ndarray, max_speed: float) -> tuple[np.ndarray, float]:
        mag = float(np.linalg.norm(move))
        if mag > max_speed:
            move = move / max(1e-8, mag) * max_speed
            mag = max_speed
        return move.astype(np.float32), mag

    def _predictive_risk(
        self,
        cell: CellState,
        move: np.ndarray,
        move_mag: float,
    ) -> float:
        cfg = self.cfg
        pos = cell.pos.copy()
        energy = float(cell.energy)
        temp = float(cell.temp)
        damage = float(cell.damage)
        forage = float(cell.traits[T_FORAGE])
        cooling_trait = float(cell.traits[T_COOLING])

        risk = 0.0
        discount = 1.0
        for _ in range(max(1, cfg.lookahead_steps)):
            pos = np.clip(pos + move, [0, 0], [cfg.width, cfg.height])
            fv = self._sample_fields(pos)
            energy = np.clip(
                energy - cfg.basal_cost - cfg.move_cost * move_mag + cfg.nutrient_gain * forage * fv["nutrient"],
                0.0,
                1.0,
            )
            cooling = cfg.cooling_rate * cooling_trait + cfg.cool_zone_cooling * fv["cool"]
            temp = np.clip(
                temp + cfg.heat_move * move_mag**2 + cfg.heat_hazard * fv["toxin"] - cooling,
                0.0,
                1.5,
            )
            overheat = max(0.0, temp - cfg.t_crit)
            repair = cfg.repair_rate * (1.0 if energy > cfg.repair_min_energy else 0.35)
            damage = np.clip(
                damage + cfg.dmg_toxin * fv["toxin"] + cfg.dmg_overheat * overheat - repair,
                0.0,
                cfg.d_max,
            )
            step_risk = (
                cfg.risk_temp_weight * max(0.0, temp - cfg.t_crit)
                + cfg.risk_damage_weight * damage
                + cfg.risk_energy_weight * max(0.0, 0.20 - energy)
            )
            # Hard penalty near fatal boundaries.
            if temp >= cfg.t_fatal or damage >= cfg.d_max or energy <= 0.0:
                step_risk += 2.5
            risk += discount * step_risk
            discount *= 0.9
        return float(risk)

    def _choose_move(
        self,
        cell: CellState,
        nut_grad: np.ndarray,
        tox_grad: np.ndarray,
        sig_grad: np.ndarray,
        local_signal: float,
    ) -> tuple[np.ndarray, float]:
        cfg = self.cfg
        forage = float(cell.traits[T_FORAGE])
        toxin_avoid = float(cell.traits[T_TOXIN_AVOID])
        speed = float(cell.traits[T_SPEED])
        signal_avoid = float(cell.traits[T_SIGNAL_AVOID])

        noise = self.rng.normal(0, 0.15, size=2).astype(np.float32)
        bias = nut_grad * (0.50 * forage) - tox_grad * (0.85 * toxin_avoid)
        if cfg.use_signaling and local_signal > cfg.signal_threshold:
            bias -= sig_grad * (0.8 * signal_avoid)

        max_speed = float(0.30 * speed)
        base_move, base_mag = self._cap_speed(bias + noise, max_speed=max_speed)
        if not cfg.predictive_steering:
            return base_move, base_mag

        candidate_1, mag_1 = self._cap_speed(bias + noise * 0.35, max_speed=max_speed)
        candidate_2, mag_2 = self._cap_speed(
            nut_grad * (0.70 * forage) - tox_grad * (1.10 * toxin_avoid) - sig_grad * (0.35 * signal_avoid),
            max_speed=max_speed,
        )
        candidate_3, mag_3 = self._cap_speed(
            -tox_grad * (1.35 * toxin_avoid) + noise * 0.20,
            max_speed=max_speed,
        )

        candidates = [
            (base_move, base_mag),
            (candidate_1, mag_1),
            (candidate_2, mag_2),
            (candidate_3, mag_3),
        ]
        scores = [self._predictive_risk(cell, mv, mm) for mv, mm in candidates]
        best_idx = int(np.argmin(scores))
        return candidates[best_idx]

    def _trait_diversity_entropy(self, alive_cells: list[CellState]) -> float:
        if not alive_cells:
            return 0.0
        # Coarse phenotype bins for macro-diversity tracking.
        sigs = []
        edges = np.array([0.75, 1.05, 1.35], dtype=np.float32)
        for c in alive_cells:
            t = c.traits
            b0 = int(np.digitize(t[T_FORAGE], edges))
            b1 = int(np.digitize(t[T_TOXIN_AVOID], edges))
            b2 = int(np.digitize(t[T_SPEED], edges))
            b3 = int(np.digitize(t[T_DIVISION], np.array([0.70, 0.80, 0.90], dtype=np.float32)))
            sigs.append((b0, b1, b2, b3))
        counts = {}
        for s in sigs:
            counts[s] = counts.get(s, 0) + 1
        ps = np.array(list(counts.values()), dtype=np.float64)
        ps /= np.sum(ps)
        h = -np.sum(ps * np.log(ps + 1e-12))
        h_max = np.log(max(2, len(counts)))
        return float(h / h_max)

    def step(self) -> dict:
        """Advance one time step."""
        assert self.fields is not None
        self.step_count += 1
        cfg = self.cfg

        self._regrow_resources()

        # Diffuse and decay signal field
        if cfg.use_signaling and self.signal_field is not None:
            self.signal_field *= cfg.signal_decay
            kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
            diffused = convolve(self.signal_field, kernel * cfg.signal_diffusion_rate, mode="constant")
            self.signal_field = np.clip(self.signal_field + diffused, 0.0, 10.0)

        new_cells = []
        births = 0
        deaths = 0
        deaths_starvation = 0
        deaths_overheating = 0
        deaths_damage = 0

        for cell in self.cells:
            if not cell.alive:
                continue

            # Sample local fields
            fv = self._sample_fields(cell.pos)
            nut_grad = self._nutrient_gradient(cell.pos)
            tox_grad = self.fields.toxin_gradient(cell.pos)

            sig_grad = np.zeros(2, dtype=np.float32)
            local_sig = 0.0

            # Signal avoidance (move away from high signal)
            if cfg.use_signaling and self.signal_field is not None:
                sig_grad = self._signal_gradient(cell.pos)
                local_sig = self._sample_signal(cell.pos)

            move, move_mag = self._choose_move(cell, nut_grad, tox_grad, sig_grad, local_sig)

            cell.pos = np.clip(
                cell.pos + move,
                [0, 0],
                [cfg.width, cfg.height],
            )
            cell.age += 1

            next_fv = self._sample_fields(cell.pos)
            forage = float(cell.traits[T_FORAGE])
            cooling_trait = float(cell.traits[T_COOLING])

            # Update internal state
            cell.energy = np.clip(
                cell.energy - cfg.basal_cost - cfg.move_cost * move_mag + cfg.nutrient_gain * forage * next_fv["nutrient"],
                0.0, 1.0,
            )
            cooling = cfg.cooling_rate * cooling_trait + cfg.cool_zone_cooling * next_fv["cool"]
            cell.temp = np.clip(
                cell.temp + cfg.heat_move * move_mag ** 2 + cfg.heat_hazard * next_fv["toxin"] - cooling,
                0.0, 1.5,
            )
            overheat = max(0.0, cell.temp - cfg.t_crit)
            move_rest = max(0.0, 1.0 - move_mag / max(1e-6, 0.30 * float(cell.traits[T_SPEED])))
            repair = cfg.repair_rate * move_rest * (1.0 if cell.energy > cfg.repair_min_energy else 0.35)
            cell.damage = np.clip(
                cell.damage + cfg.dmg_toxin * next_fv["toxin"] + cfg.dmg_overheat * overheat - repair,
                0.0, cfg.d_max,
            )
            self._consume_nutrients(
                cell.pos,
                amount=cfg.nutrient_consumption * forage * np.clip(next_fv["nutrient"], 0.0, 1.5),
            )

            # Secrete signal when stressed
            if cfg.use_signaling and self.signal_field is not None:
                stress = cell.damage + max(0, cell.temp - cfg.t_crit)
                cell.stress_memory = 0.9 * cell.stress_memory + 0.1 * stress
                if stress > 0.2:
                    emit_gain = float(cell.traits[T_SIGNAL_EMIT])
                    self._deposit_signal(cell.pos, stress * 0.5 * emit_gain)

            # Check death
            cause = ""
            if cell.energy <= 0:
                cause = "starvation"
            elif cell.temp >= cfg.t_fatal:
                cause = "overheating"
            elif cell.damage >= cfg.d_max:
                cause = "damage_overload"
            if cause:
                cell.alive = False
                cell.death_cause = cause
                deaths += 1
                if cause == "starvation":
                    deaths_starvation += 1
                elif cause == "overheating":
                    deaths_overheating += 1
                elif cause == "damage_overload":
                    deaths_damage += 1
                continue

            # Division
            div_threshold = np.clip(
                cfg.division_energy + (float(cell.traits[T_DIVISION]) - float(TRAIT_BASE[T_DIVISION])),
                0.55,
                0.98,
            )
            div_threshold += float(self.rng.normal(0.0, cfg.division_jitter))
            div_threshold = float(np.clip(div_threshold, 0.50, 0.99))
            if (
                cell.energy > div_threshold
                and cell.age >= cfg.min_division_age
                and sum(int(c.alive) for c in self.cells) + len(new_cells) < cfg.max_cells
            ):
                available = max(0.0, cell.energy - cfg.division_cost)
                if available <= 0.2:
                    continue
                parent_id = cell.cell_id
                child_id = self._new_cell_id()
                child_traits = self._mutate_traits(cell.traits)
                child_energy = available * 0.5
                cell.energy = available * 0.5
                offset = self.rng.normal(0, 0.5, size=2).astype(np.float32)
                child_pos = np.clip(cell.pos + offset, [0, 0], [cfg.width, cfg.height])
                new_cells.append(CellState(
                    cell_id=child_id,
                    lineage_id=cell.lineage_id,
                    parent_id=parent_id,
                    pos=child_pos,
                    energy=child_energy,
                    temp=cell.temp,
                    damage=0.0,
                    traits=child_traits,
                    generation=cell.generation + 1,
                ))
                births += 1

        self.cells.extend(new_cells)
        alive_cells = [c for c in self.cells if c.alive]
        self._births_total += births
        self._deaths_total += deaths
        self._deaths_by_cause["starvation"] += deaths_starvation
        self._deaths_by_cause["overheating"] += deaths_overheating
        self._deaths_by_cause["damage_overload"] += deaths_damage

        if alive_cells:
            traits = np.stack([c.traits for c in alive_cells], axis=0)
            mean_traits = np.mean(traits, axis=0)
            max_gen = int(max(c.generation for c in alive_cells))
            lineages_alive = int(len({c.lineage_id for c in alive_cells}))
        else:
            mean_traits = TRAIT_BASE.copy()
            max_gen = 0
            lineages_alive = 0

        # Record history
        record = {
            "step": self.step_count,
            "population": len(alive_cells),
            "mean_energy": np.mean([c.energy for c in alive_cells]) if alive_cells else 0,
            "mean_damage": np.mean([c.damage for c in alive_cells]) if alive_cells else 0,
            "mean_temp": np.mean([c.temp for c in alive_cells]) if alive_cells else 0,
            "births": births,
            "deaths": deaths,
            "deaths_starvation": deaths_starvation,
            "deaths_overheating": deaths_overheating,
            "deaths_damage": deaths_damage,
            "births_total": self._births_total,
            "deaths_total": self._deaths_total,
            "lineages_alive": lineages_alive,
            "max_generation": max_gen,
            "diversity_entropy": self._trait_diversity_entropy(alive_cells),
            "signal_mean": float(np.mean(self.signal_field)) if self.signal_field is not None else 0.0,
            "nutrient_stock_mean": float(np.mean(self.nutrient_stock)) if self.nutrient_stock is not None else 1.0,
            "trait_forage": float(mean_traits[T_FORAGE]),
            "trait_toxin_avoid": float(mean_traits[T_TOXIN_AVOID]),
            "trait_speed": float(mean_traits[T_SPEED]),
            "trait_cooling": float(mean_traits[T_COOLING]),
            "trait_division": float(mean_traits[T_DIVISION]),
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

    def snapshot(self) -> dict:
        """Return a render-friendly snapshot of the current ecosystem state."""
        assert self.fields is not None
        alive = [c for c in self.cells if c.alive]
        cells = [
            {
                "x": float(c.pos[0]),
                "y": float(c.pos[1]),
                "energy": float(c.energy),
                "temp": float(c.temp),
                "damage": float(c.damage),
                "generation": int(c.generation),
                "lineage_id": int(c.lineage_id),
                "forage": float(c.traits[T_FORAGE]),
                "toxin_avoid": float(c.traits[T_TOXIN_AVOID]),
                "speed": float(c.traits[T_SPEED]),
            }
            for c in alive
        ]
        nutrients = []
        for i, src in enumerate(self.fields.nutrients):
            stock = float(self.nutrient_stock[i]) if self.nutrient_stock is not None else 1.0
            nutrients.append({
                "x": float(src.center[0]),
                "y": float(src.center[1]),
                "radius": float(src.radius),
                "stock": stock,
            })
        toxins = [
            {"x": float(src.center[0]), "y": float(src.center[1]), "radius": float(src.radius)}
            for src in self.fields.toxins
        ]
        cool = [
            {"x": float(src.center[0]), "y": float(src.center[1]), "radius": float(src.radius)}
            for src in self.fields.cool_zones
        ]
        return {
            "step": int(self.step_count),
            "cells": cells,
            "nutrients": nutrients,
            "toxins": toxins,
            "cool_zones": cool,
            "population": len(alive),
            "history": self.history[-1] if self.history else {},
        }

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
