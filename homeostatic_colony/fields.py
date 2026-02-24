"""
Spatial fields for the 2D world: nutrients, toxins, and cool zones.

Each field source is a circular region with a strength that decays
with distance from the center (Gaussian-like falloff).
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class FieldSource:
    """A single circular source in the world."""
    center: np.ndarray  # shape (2,)
    radius: float
    strength: float
    active: bool = True
    respawn_counter: int = 0

    def intensity_at(self, pos: np.ndarray) -> float:
        """Return field intensity at a given position (smooth Gaussian falloff)."""
        if not self.active:
            return 0.0
        dist = np.linalg.norm(pos - self.center)
        if dist > self.radius * 2.5:
            return 0.0
        return float(self.strength * np.exp(-0.5 * (dist / self.radius) ** 2))


class FieldManager:
    """Manages all spatial fields in the world."""

    def __init__(
        self,
        width: float,
        height: float,
        rng: np.random.Generator,
    ):
        self.width = width
        self.height = height
        self.rng = rng
        self.nutrients: list[FieldSource] = []
        self.toxins: list[FieldSource] = []
        self.cool_zones: list[FieldSource] = []

    def generate(
        self,
        n_nutrients: int,
        n_toxins: int,
        n_cool: int,
        nutrient_radius: float,
        toxin_radius: float,
        cool_radius: float,
        nutrient_strength: float,
        toxin_strength: float,
        cool_strength: float,
    ) -> None:
        """Randomly place all field sources."""
        self.nutrients = self._make_sources(n_nutrients, nutrient_radius, nutrient_strength)
        self.toxins = self._make_sources(n_toxins, toxin_radius, toxin_strength)
        self.cool_zones = self._make_sources(n_cool, cool_radius, cool_strength)

    def _make_sources(self, n: int, radius: float, strength: float) -> list[FieldSource]:
        margin = radius
        sources = []
        for _ in range(n):
            cx = self.rng.uniform(margin, self.width - margin)
            cy = self.rng.uniform(margin, self.height - margin)
            sources.append(FieldSource(
                center=np.array([cx, cy], dtype=np.float32),
                radius=radius,
                strength=strength,
            ))
        return sources

    def sample_at(self, pos: np.ndarray) -> dict[str, float]:
        """Return nutrient, toxin, and cool intensities at a position."""
        nutrient = sum(s.intensity_at(pos) for s in self.nutrients)
        toxin = sum(s.intensity_at(pos) for s in self.toxins)
        cool = sum(s.intensity_at(pos) for s in self.cool_zones)
        return {"nutrient": nutrient, "toxin": toxin, "cool": cool}

    def nutrient_gradient(self, pos: np.ndarray, eps: float = 0.3) -> np.ndarray:
        """Finite-difference gradient of nutrient field at pos."""
        grad = np.zeros(2, dtype=np.float32)
        for dim in range(2):
            pos_plus = pos.copy()
            pos_minus = pos.copy()
            pos_plus[dim] += eps
            pos_minus[dim] -= eps
            val_plus = sum(s.intensity_at(pos_plus) for s in self.nutrients)
            val_minus = sum(s.intensity_at(pos_minus) for s in self.nutrients)
            grad[dim] = (val_plus - val_minus) / (2 * eps)
        return grad

    def toxin_gradient(self, pos: np.ndarray, eps: float = 0.3) -> np.ndarray:
        """Finite-difference gradient of toxin field at pos."""
        grad = np.zeros(2, dtype=np.float32)
        for dim in range(2):
            pos_plus = pos.copy()
            pos_minus = pos.copy()
            pos_plus[dim] += eps
            pos_minus[dim] -= eps
            val_plus = sum(s.intensity_at(pos_plus) for s in self.toxins)
            val_minus = sum(s.intensity_at(pos_minus) for s in self.toxins)
            grad[dim] = (val_plus - val_minus) / (2 * eps)
        return grad

    def update_respawns(self, respawn_steps: int) -> None:
        """Tick respawn counters for depleted nutrients."""
        for s in self.nutrients:
            if not s.active:
                s.respawn_counter += 1
                if s.respawn_counter >= respawn_steps:
                    s.active = True
                    s.respawn_counter = 0

    def deplete_nutrient(self, idx: int) -> None:
        """Mark a nutrient source as temporarily depleted."""
        if 0 <= idx < len(self.nutrients):
            self.nutrients[idx].active = False
            self.nutrients[idx].respawn_counter = 0

    def render_field_image(self, resolution: int = 100) -> dict[str, np.ndarray]:
        """Render field intensities as 2D arrays for visualization."""
        xs = np.linspace(0, self.width, resolution)
        ys = np.linspace(0, self.height, resolution)
        nutrient_map = np.zeros((resolution, resolution), dtype=np.float32)
        toxin_map = np.zeros((resolution, resolution), dtype=np.float32)
        cool_map = np.zeros((resolution, resolution), dtype=np.float32)

        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                pos = np.array([x, y], dtype=np.float32)
                vals = self.sample_at(pos)
                nutrient_map[i, j] = vals["nutrient"]
                toxin_map[i, j] = vals["toxin"]
                cool_map[i, j] = vals["cool"]

        return {"nutrient": nutrient_map, "toxin": toxin_map, "cool": cool_map}
