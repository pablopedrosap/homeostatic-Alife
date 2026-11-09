"""Tests for the HAB-P7 body schema and central governor (synthetic bodies with known physics)."""

import numpy as np
import pytest

from homeostatic_colony.control.central_governor import (
    BodySchema, CentralGovernor, sense_temperature, wear_rate,
)

DT = 0.02


def synthetic_walk(gain=0.57, tau=10.0, load=1.0, seed=0, speeds=(0.4, 0.6, 0.8, 1.0, 1.2), block_s=5.0, passes=3):
    """A body whose heat input per motor is q_i = load · k_i · (0.05 + 0.1 v²) (+ noise)."""
    rng = np.random.default_rng(seed)
    k = np.linspace(0.6, 1.6, 12)
    T = rng.uniform(0, 0.3, 12)
    Ts, Qs, Vs = [sense_temperature(T)], [np.zeros(12)], [speeds[0]]
    for _ in range(passes):
        for v in rng.permutation(speeds):
            for _ in range(int(block_s / DT)):
                q = load * k * (0.05 + 0.1 * v * v) * (1 + 0.1 * rng.standard_normal(12))
                T = np.maximum(0, T + DT * (gain * q - T / tau))
                Ts.append(sense_temperature(T)); Qs.append(q); Vs.append(v)
    return np.asarray(Ts), np.asarray(Qs), np.asarray(Vs), k


def test_body_schema_recovers_own_physics_from_quantised_readings():
    T, Q, V, k = synthetic_walk()
    s = BodySchema.fit(T, Q, V, DT)
    assert s.gain == pytest.approx(0.57, rel=0.08)
    assert s.tau == pytest.approx(10.0, rel=0.08)
    # heat-vs-speed shape: hottest motor at 1.2 m/s ≈ 1.6·(0.05 + 0.144)
    assert s.q_nominal(1.2).max() == pytest.approx(1.6 * (0.05 + 0.1 * 1.44), rel=0.1)


def _schema():
    coef = np.zeros((12, 3))
    k = np.linspace(0.6, 1.6, 12)
    coef[:, 0], coef[:, 2] = 0.05 * k, 0.1 * k
    return BodySchema(gain=0.57, tau=10.0, coef=coef)


def test_governor_paces_to_keep_end_temperature_under_threshold():
    s = _schema()
    gov = CentralGovernor(s, mission_s=180, dt=DT, t_limit=0.7)
    v = gov.update(np.full(12, 0.2), s.q_nominal(gov.cmd))
    t_ss = s.gain * s.tau * s.q_nominal(v)
    assert t_ss.max() <= 0.7 + 1e-9
    faster = min(v + 0.05, 1.2)
    if faster > v:
        assert (s.gain * s.tau * s.q_nominal(faster)).max() > 0.7


def test_heavier_body_is_detected_online_and_slows_the_pace():
    s = _schema()
    paces = {}
    for load, adapt in ((1.0, True), (1.6, True), (1.6, False)):
        gov = CentralGovernor(s, mission_s=180, dt=DT, adapt=adapt)
        T = np.full(12, 0.2)
        for _ in range(int(20 / DT)):          # 20 s of walking
            q = load * s.q_nominal(gov.cmd)
            cmd = gov.update(sense_temperature(T), q)
            T = T + DT * (s.gain * load * s.q_nominal(cmd) - T / s.tau)
        paces[(load, adapt)] = gov.cmd
    assert paces[(1.6, True)] < paces[(1.0, True)]          # knows it's carrying more → slower
    assert paces[(1.6, False)] > paces[(1.6, True)]         # without self-calibration it doesn't know


def test_end_spurt_near_the_finish():
    """Same body state: early in the mission it must hold back; 2 s from the end it can sprint."""
    s = _schema()
    temps = np.full(12, 0.3)
    early = CentralGovernor(s, mission_s=180, dt=DT, t_limit=0.7)
    v_early = early.update(temps, np.zeros(12))
    late = CentralGovernor(s, mission_s=180, dt=DT, t_limit=0.7)
    late.steps = int(178 / DT) // 50 * 50                   # 2 s left, on a planning step
    v_late = late.update(temps, s.q_nominal(late.cmd))      # nominal body: heat as predicted
    assert v_early < 1.0
    assert v_late == pytest.approx(1.2)


def test_wear_doubles_every_10_degrees():
    assert wear_rate(np.array([10 / 75]))[0] == pytest.approx(2.0)
    assert wear_rate(np.array([0.0]))[0] == pytest.approx(1.0)


def test_anticipatory_governor_precools_before_a_short_heavy_burst():
    """A short burst (comparable to the thermal time constant): arriving cool lowers the peak,
    so a governor that knows the burst is coming walks slower beforehand."""
    from homeostatic_colony.control.anticipatory import AnticipatoryGovernor
    s = _schema()
    heavy = {6.0: np.full(12, 2.5)}
    T0 = np.full(12, 0.3)
    knows = AnticipatoryGovernor(s, 180.0, [(0, 40, 0.0), (40, 55, 6.0), (55, 180, 0.0)], heavy, dt=DT)
    naive = AnticipatoryGovernor(s, 180.0, [(0, 40, 0.0), (40, 55, 0.0), (55, 180, 0.0)], heavy, dt=DT)  # same phases, no load
    knows.steps = naive.steps = int(30 / DT) // 50 * 50      # 10 s before the burst
    v_knows = knows.update(T0, s.q_nominal(knows.cmd))
    v_naive = naive.update(T0, s.q_nominal(naive.cmd))
    assert v_knows < v_naive


def test_long_load_makes_precooling_pointless():
    """A load much longer than the thermal time constant: the peak no longer depends on the entry
    temperature, so the governor does not sacrifice speed beforehand (physics, not a bug)."""
    from homeostatic_colony.control.anticipatory import AnticipatoryGovernor
    s = _schema()
    gov = AnticipatoryGovernor(s, 180.0, [(0, 60, 0.0), (60, 120, 6.0), (120, 180, 0.0)],
                               {6.0: np.full(12, 2.5)}, dt=DT)
    peaks = gov._simulate_peaks(np.full(12, 0.3), 40.0)
    assert peaks.max() - peaks.min() < 0.1


def test_anticipatory_governor_is_vectorised_and_bounded():
    from homeostatic_colony.control.anticipatory import AnticipatoryGovernor
    s = _schema()
    gov = AnticipatoryGovernor(s, 180.0, [(0, 90, 0.0), (90, 180, 3.0)], {3.0: np.full(12, 1.7)}, dt=DT)
    peaks = gov._simulate_peaks(np.full(12, 0.2), 0.0)
    assert peaks.shape == (len(gov.candidates),)
    assert np.all(np.diff(peaks) <= 1e-9)            # candidates fastest-first → peaks non-increasing
    v = gov.update(np.full(12, 0.2), np.zeros(12))
    assert 0.4 <= v <= 1.2
