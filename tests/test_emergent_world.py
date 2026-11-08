"""Tests for the no-reward emergent world, incl. the HAB-E2v weather / yoked-control extensions."""

import platform

import numpy as np
import pytest

from homeostatic_colony.alife.emergent_world import (
    N_IN, EmergentWorld, WorldConfig, k_selection_config,
)

# Golden values were recorded on macOS arm64. float32 rounding differs slightly across CPU
# architectures and a 400-step population simulation is chaotic, so elsewhere we allow 1%.
GOLDEN_PLATFORM = platform.system() == "Darwin" and platform.machine() == "arm64"


@pytest.mark.parametrize("cfg, births, pop", [
    (WorldConfig(seed=3), 3993, 500),
    (k_selection_config(seed=3), 306, 22),
    (WorldConfig(seed=3, blind_interoception=True), 1304, 500),
])
def test_frozen_world_golden_values(cfg, births, pop):
    """The original (constant, all-self-sensing) worlds must never drift: they are the
    k = 0 control of HAB-E2v and the substrate of the published HAB-E2 result."""
    w = EmergentWorld(cfg)
    w.run(400)
    if GOLDEN_PLATFORM:
        assert w.births_total == births
        assert int(w.alive.sum()) == pop
    else:
        assert w.births_total == pytest.approx(births, rel=0.01)
        assert int(w.alive.sum()) == pytest.approx(pop, rel=0.05, abs=2)


def _small(**kw):
    base = dict(seed=0, n_initial=40, capacity=80)
    base.update(kw)
    return WorldConfig(**base)


def test_weather_is_zero_mean_unit_variance_ou():
    w = EmergentWorld(_small(weather_amp=0.01, weather_tau=50.0))
    w.reset()
    ws = np.array([w.step()["weather"] for _ in range(6000)])
    assert abs(ws.mean()) < 0.35
    assert 0.6 < ws.std() < 1.4
    lag = 50
    ac = np.corrcoef(ws[:-lag], ws[lag:])[0, 1]
    assert 0.15 < ac < 0.6          # exp(-1) ≈ 0.37 at one correlation time


def test_no_weather_when_amplitude_zero():
    w = EmergentWorld(_small())
    w.reset()
    assert all(w.step()["weather"] == 0.0 for _ in range(50))


def test_weather_heats_population():
    """Same seed, a frozen +1 heatwave vs none: the load must add amp per step to T.

    Blind organisms + few steps keep behaviour identical between the two worlds (no
    temperature feedback into movement, nobody old enough to divide or hot enough to die),
    so the difference is pure physics.
    """
    temps = {}
    for amp in (0.0, 0.02):
        w = EmergentWorld(_small(weather_amp=amp, weather_tau=1e9, blind_interoception=True))
        w.reset()
        w.weather = 1.0  # tau -> inf: the OU process stays (essentially) frozen at +1
        for _ in range(5):
            r = w.step()
        temps[amp] = r["mean_T"]
    assert temps[0.02] - temps[0.0] == pytest.approx(5 * 0.02, abs=0.01)


def test_founders_are_exactly_balanced():
    w = EmergentWorld(_small(sense_frac_init=0.5))
    w.reset()
    assert w.senses[:40].sum() == 20


def test_sensing_trait_is_inherited_without_mutation():
    w = EmergentWorld(_small(sense_frac_init=0.5))
    w.reset()
    founder_trait = w.senses[:40].copy()
    for _ in range(300):
        if w.step()["population"] == 0:
            break
    assert w.births_total > 0
    alive = np.where(w.alive)[0]
    assert np.array_equal(w.senses[alive], founder_trait[w.lineage[alive]])


def test_yoked_organisms_never_read_their_own_body():
    w = EmergentWorld(_small(sense_frac_init=0.0))
    w.reset()
    for _ in range(100):
        w.step()
        y = np.where(w.alive & ~w.senses)[0]
        # partners are reassigned at the start of the next step; check the live invariant
        w._assign_partners(w.alive & ~w.senses)
        if y.size > 1:
            assert np.all(w.partner[y] != y)
            assert np.all(w.alive[w.partner[y]])


def test_probe_is_an_intervention_on_felt_temperature():
    w = EmergentWorld(_small())
    w.reset()
    i = 0
    w.p_W1[i] = 0.0
    w.p_W2[i] = 0.0
    w.b1[i] = 0.0
    w.b2[i] = 0.0
    w.p_W1[i, 0, 6] = 2.0      # hidden 0 <- felt temperature
    w.p_W2[i, 0, 0] = 2.0      # move x <- hidden 0
    temps = np.linspace(0.0, 1.0, 6)
    speeds = w.probe_thermal_reflex(np.array([i]), temps)
    assert speeds.shape == (1, 6)
    assert np.all(np.diff(speeds[0]) >= 0) and speeds[0, -1] > speeds[0, 0]
    assert np.all(speeds <= w.cfg.max_speed + 1e-6)
    # cooling input is input 4 and is silenced by the thermal probe
    w.p_W1[i, 0, 6] = 0.0
    w.p_W1[i, 0, 4] = -2.0
    flat = w.probe_thermal_reflex(np.array([i]), temps)
    assert np.allclose(flat, flat[0, 0])
    kin = w.probe_speed(np.array([i]), 0.9, np.linspace(0, 1, 5))
    assert kin[0, -1] > kin[0, 0]
    assert N_IN == 8
