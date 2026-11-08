"""Emergent homeostasis via pure selection — no reward, no gradient, no fitness function.

This is the honest ALife substrate (see docs/EMERGENCE.md). Nothing optimizes anything:

  * Each organism carries a small neural controller. Its **genome** = the controller's
    birth weights, inherited from a parent with mutation. We do NOT author behavior.
  * Within its lifetime the controller **rewires via a local Hebbian (Oja) rule** — driven
    only by its own pre/post activations, never by any reward or error signal.
  * Physics has real consequences: run fast → overheat, stop foraging → starve, touch
    toxins → damage. An organism whose internal state leaves the viable region is removed.
  * Reproduction is a **physical event**: enough energy to split → it splits. There is no
    fitness score deciding who reproduces.

Homeostasis, if it appears, is not "learned to maximize" anything. It is simply the
regulation strategy left standing after the non-persisters are gone — the way life's own
setpoints arose. The measurement question (docs/EMERGENCE.md, HAB-E2) is falsifiable: does
the living population converge, unbidden, onto a viable internal region no one specified?

Darwinian, not Lamarckian: offspring inherit the parent's *genome* (birth weights), not the
weights the parent drifted to in life. That lets lifetime plasticity guide evolution (Baldwin
effect) without directly writing acquired changes back into the germline.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Observation layout (what an organism senses): local nutrient gradient (2), local toxin
# gradient (2), local cool intensity (1), and interoception E/T/D (3) = 8 inputs.
N_IN = 8
N_OUT = 2  # movement vector


@dataclass
class WorldConfig:
    """Fixed, pre-registered world constants. Set so that life is *possible*, then frozen.

    These describe the world's physics, NOT the organism's drives. They are never tuned to
    produce a particular outcome (see docs/EMERGENCE.md anti-rigging rule).
    """
    width: float = 30.0
    height: float = 30.0
    n_nutrients: int = 10
    n_toxins: int = 5
    n_cool: int = 4
    nutrient_radius: float = 3.2
    toxin_radius: float = 3.2
    cool_radius: float = 2.6

    # Metabolism / physics
    basal_cost: float = 0.004        # energy burned per step just by existing
    move_cost: float = 0.010         # energy per unit speed
    nutrient_gain: float = 0.13      # energy gained from local nutrient
    heat_speed: float = 0.09         # SPEED -> heat (the run-fast-or-cool dilemma)
    heat_hazard: float = 0.05        # toxin -> heat
    cooling_rate: float = 0.016      # passive cooling per step
    cool_zone_cooling: float = 0.03  # extra cooling in cool zones
    dmg_toxin: float = 0.035
    dmg_overheat: float = 0.05
    repair_rate: float = 0.010       # damage repaired per step when moving slowly + fed
    repair_min_energy: float = 0.30

    t_crit: float = 0.7              # overheating onset (damage begins)
    t_fatal: float = 1.2             # death by heat
    d_max: float = 1.0               # death by damage
    max_speed: float = 0.35

    # Life history
    e_init: float = 0.60
    t_init: float = 0.30
    capacity: int = 500              # hard array capacity
    n_initial: int = 150
    division_energy: float = 0.78    # split when energy exceeds this (a physical threshold)
    division_cost: float = 0.30      # energy consumed by the act of dividing
    min_division_age: int = 25
    repro_cooldown: int = 0          # min steps between an organism's births (rate-limits r)

    # Resource ecology (K-selection): nutrients deplete when eaten and regrow slowly, so
    # energy becomes a genuinely scarce, competitive resource. Off by default (r-selection).
    scarce_resources: bool = False
    nutrient_regrowth: float = 0.02
    nutrient_consumption: float = 0.15

    # Controller / heredity / plasticity
    n_hidden: int = 8
    weight_init_std: float = 0.5
    mutation_std: float = 0.08       # genome mutation on inheritance
    weight_clip: float = 3.0
    hebb_eta: float = 0.02           # within-lifetime plasticity rate (0 = no learning)

    max_steps: int = 4000
    seed: int = 0

    # Ablation switches (for falsifiable tests — see docs/EMERGENCE.md HAB-E2).
    blind_interoception: bool = False   # zero the E/T/D inputs: can't sense own state
    # (set hebb_eta = 0.0 to ablate within-lifetime plasticity)

    # Environmental variability (HAB-E2v, docs/prereg/HAB_E2v_regulator_law.md). A global,
    # zero-mean thermal load weather_amp * w(t), where w is a unit-variance Ornstein-Uhlenbeck
    # process: the average climate is unchanged, only its volatility. 0 = the constant world.
    weather_amp: float = 0.0
    weather_tau: float = 50.0

    # Heritable interoception trait (HAB-E2v). Founders that sense their OWN E/T/D; the rest
    # are "yoked" — they receive a partner organism's E/T/D (matched input statistics, zero
    # self-information). Inherited by offspring, never mutates. 1.0 = everyone self-senses.
    sense_frac_init: float = 1.0
    # Which interoceptive channels (0=E, 1=T, 2=D) a yoked organism reads from its partner;
    # the others stay its own. Exploratory channel dissection — default yokes all three.
    yoke_channels: tuple[int, ...] = (0, 1, 2)


def k_selection_config(seed: int = 0, **overrides) -> "WorldConfig":
    """A K-selection regime: reproduction is hard and requires a long, well-regulated life.

    Structural (not tuned) changes vs the default r-selection world:
      * scarce, depletable, slowly-regrowing nutrients -> energy is competitive
      * reproduction demands high accumulated energy and is expensive
      * long maturation + a long refractory period -> you cannot out-breed via speed
    Under these, the disposable-soma "run hot, die young, breed fast" strategy should stop
    paying, and self-maintenance should become the winning strategy. Set once; frozen.
    """
    # Values calibrated ONLY for life-possibility (population persists + reproduces),
    # blind to whether regulation wins. Scarcity is density-dependent: resources start full
    # and the population's own consumption depletes them, so competition emerges endogenously
    # as carrying capacity (~170, well below the 500 array cap) is approached.
    cfg = WorldConfig(
        seed=seed,
        scarce_resources=True,
        nutrient_regrowth=0.10,      # regrows, but finite -> competitive
        nutrient_consumption=0.015,  # depletion accumulates with population density
        n_nutrients=20,
        division_energy=0.80,        # must accumulate substantial energy to reproduce
        division_cost=0.50,          # reproduction is expensive (parent drops to ~0.15 E)
        min_division_age=40,         # must survive to maturity first
        repro_cooldown=40,           # refractory period -> cannot out-breed via raw speed
        n_initial=150,
    )
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


class _VecFields:
    """Vectorized Gaussian field sampler (nutrients / toxins / cool)."""

    def __init__(self, cfg: WorldConfig, rng: np.random.Generator):
        def place(n, r):
            c = np.stack([rng.uniform(r, cfg.width - r, size=n),
                          rng.uniform(r, cfg.height - r, size=n)], axis=1)
            return c.astype(np.float32), np.full(n, r, np.float32)
        self.nc, self.nr = place(cfg.n_nutrients, cfg.nutrient_radius)
        self.tc, self.tr = place(cfg.n_toxins, cfg.toxin_radius)
        self.cc, self.cr = place(cfg.n_cool, cfg.cool_radius)
        # Per-source nutrient stock (1 = full). Only used in scarce (K-selection) mode.
        self.stock = np.ones(len(self.nr), np.float32)

    @staticmethod
    def _matrix(pos, centers, radii):
        # pos [N,2], centers [S,2] -> per-source intensity [N,S]
        d2 = ((pos[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        return np.exp(-0.5 * d2 / (radii[None, :] ** 2)).astype(np.float32)

    @staticmethod
    def _intensity(pos, centers, radii):
        return _VecFields._matrix(pos, centers, radii).sum(1)

    def nutrient_matrix(self, pos):
        return self._matrix(pos, self.nc, self.nr)   # [N,S]

    def nutrient_scalar(self, pos, use_stock):
        M = self._matrix(pos, self.nc, self.nr)
        return (M * self.stock[None, :]).sum(1) if use_stock else M.sum(1)

    def sample(self, pos):
        return (self._intensity(pos, self.nc, self.nr),
                self._intensity(pos, self.tc, self.tr),
                self._intensity(pos, self.cc, self.cr))

    def gradient(self, pos, centers, radii, eps=0.3):
        g = np.zeros_like(pos)
        for dim in range(2):
            pp = pos.copy(); pp[:, dim] += eps
            pm = pos.copy(); pm[:, dim] -= eps
            g[:, dim] = (self._intensity(pp, centers, radii)
                         - self._intensity(pm, centers, radii)) / (2 * eps)
        return g

    def nutrient_gradient(self, pos, use_stock, eps=0.3):
        g = np.zeros_like(pos)
        for dim in range(2):
            pp = pos.copy(); pp[:, dim] += eps
            pm = pos.copy(); pm[:, dim] -= eps
            g[:, dim] = (self.nutrient_scalar(pp, use_stock)
                         - self.nutrient_scalar(pm, use_stock)) / (2 * eps)
        return g


class EmergentWorld:
    """Population of neural organisms under pure selection. No reward anywhere."""

    def __init__(self, cfg: WorldConfig | None = None):
        self.cfg = cfg or WorldConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        C, H = self.cfg.capacity, self.cfg.n_hidden
        # Per-organism state
        self.pos = np.zeros((C, 2), np.float32)
        self.E = np.zeros(C, np.float32)
        self.T = np.zeros(C, np.float32)
        self.D = np.zeros(C, np.float32)
        self.age = np.zeros(C, np.int32)
        self.gen = np.zeros(C, np.int32)
        self.lineage = np.zeros(C, np.int32)
        self.alive = np.zeros(C, bool)
        self.last_repro = np.zeros(C, np.int32)
        self.senses = np.ones(C, bool)          # heritable: senses own E/T/D (else yoked)
        self.partner = np.arange(C)             # yoked organisms read this one's E/T/D
        self.weather = 0.0
        # Genome (birth weights) and plastic (in-life) weights
        self.g_W1 = np.zeros((C, H, N_IN), np.float32)
        self.g_W2 = np.zeros((C, N_OUT, H), np.float32)
        self.b1 = np.zeros((C, H), np.float32)
        self.b2 = np.zeros((C, N_OUT), np.float32)
        self.p_W1 = np.zeros((C, H, N_IN), np.float32)
        self.p_W2 = np.zeros((C, N_OUT, H), np.float32)
        self.fields: _VecFields | None = None
        self.step_count = 0
        self._next_lineage = 0
        self.history: list[dict] = []
        self.births_total = 0
        self.deaths_total = 0
        self._lifespan_sum = 0.0
        self._lifespan_count = 0

    @property
    def mean_lifespan(self) -> float:
        return self._lifespan_sum / self._lifespan_count if self._lifespan_count else 0.0

    # -- helpers -----------------------------------------------------------
    def _spawn(self, idx, genome, lineage, gen, senses=True):
        cfg = self.cfg
        self.senses[idx] = senses
        self.partner[idx] = idx  # yoked newborns get a real partner on their first step
        g_W1, g_W2, b1, b2 = genome
        self.g_W1[idx], self.g_W2[idx], self.b1[idx], self.b2[idx] = g_W1, g_W2, b1, b2
        self.p_W1[idx], self.p_W2[idx] = g_W1.copy(), g_W2.copy()  # life starts from genome
        self.pos[idx] = self.rng.uniform([1, 1], [cfg.width - 1, cfg.height - 1]).astype(np.float32)
        self.E[idx] = cfg.e_init
        self.T[idx] = cfg.t_init
        self.D[idx] = 0.0
        self.age[idx] = 0
        self.gen[idx] = gen
        self.lineage[idx] = lineage
        self.alive[idx] = True
        self.last_repro[idx] = self.step_count

    def reset(self):
        cfg = self.cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.fields = _VecFields(cfg, self.rng)
        self.alive[:] = False
        self.step_count = 0
        self.history = []
        self.births_total = self.deaths_total = 0
        self._lifespan_sum = 0.0
        self._lifespan_count = 0
        self._next_lineage = 0
        self.weather = 0.0
        H = cfg.n_hidden
        # Exactly balanced founder split; the rng is only touched for mixed populations so
        # the default world reproduces its pre-HAB-E2v trajectories bit for bit.
        sensing = np.ones(cfg.n_initial, bool)
        if cfg.sense_frac_init < 1.0:
            n_sense = int(round(cfg.sense_frac_init * cfg.n_initial))
            sensing = np.zeros(cfg.n_initial, bool)
            sensing[self.rng.permutation(cfg.n_initial)[:n_sense]] = True
        for i in range(cfg.n_initial):
            genome = (
                self.rng.normal(0, cfg.weight_init_std, (H, N_IN)).astype(np.float32),
                self.rng.normal(0, cfg.weight_init_std, (N_OUT, H)).astype(np.float32),
                self.rng.normal(0, 0.1, H).astype(np.float32),
                self.rng.normal(0, 0.1, N_OUT).astype(np.float32),
            )
            self._spawn(i, genome, self._next_lineage, 0, senses=bool(sensing[i]))
            self._next_lineage += 1

    def _assign_partners(self, yoked: np.ndarray) -> None:
        """Give every yoked organism whose partner is gone (or itself) a living partner."""
        C = self.alive.shape[0]
        bad = yoked & (~self.alive[self.partner] | (self.partner == np.arange(C)))
        if not bad.any():
            return
        living = np.where(self.alive)[0]
        if living.size < 2:
            return
        bad_idx = np.where(bad)[0]
        pick = self.rng.integers(0, living.size, bad_idx.size)
        clash = living[pick] == bad_idx
        pick[clash] = (pick[clash] + 1) % living.size   # never yoke to yourself
        self.partner[bad_idx] = living[pick]

    # -- one tick ----------------------------------------------------------
    def step(self) -> dict:
        cfg = self.cfg
        assert self.fields is not None
        self.step_count += 1
        a = self.alive
        if not a.any():
            rec = {"step": self.step_count, "population": 0}
            self.history.append(rec)
            return rec

        # Weather: exact OU update, stationary unit variance, correlation time weather_tau.
        heat_load = 0.0
        if cfg.weather_amp > 0:
            rho = np.exp(-1.0 / cfg.weather_tau)
            self.weather = rho * self.weather + np.sqrt(1.0 - rho * rho) * self.rng.normal()
            heat_load = cfg.weather_amp * self.weather

        pos = self.pos
        ng = self.fields.nutrient_gradient(pos, use_stock=cfg.scarce_resources)
        tg = self.fields.gradient(pos, self.fields.tc, self.fields.tr)
        _, tox, cool = self.fields.sample(pos)

        # Interoception is normalized; gradients are scaled to a sane range.
        if cfg.blind_interoception:
            intero = np.zeros((self.E.shape[0], 3), np.float32)  # cannot sense own E/T/D
        else:
            own = np.stack([self.E, self.T, self.D], axis=1)
            intero = own
            yoked = a & ~self.senses
            if yoked.any():
                # Yoked control: a living partner's body state — same statistics, no self.
                self._assign_partners(yoked)
                intero = own.copy()
                ch = list(cfg.yoke_channels)
                intero[np.ix_(yoked, ch)] = own[np.ix_(self.partner[yoked], ch)]
        obs = np.concatenate([
            np.clip(ng, -1, 1), np.clip(tg, -1, 1),
            cool[:, None], intero,
        ], axis=1).astype(np.float32)  # [C, N_IN]

        # Forward pass with each organism's own plastic weights.
        h = np.tanh(np.einsum('nhi,ni->nh', self.p_W1, obs) + self.b1)      # [C,H]
        mv = np.tanh(np.einsum('noh,nh->no', self.p_W2, h) + self.b2)       # [C,2]
        speed = np.linalg.norm(mv, axis=1)                                  # [C]
        scale = np.minimum(1.0, cfg.max_speed / np.maximum(speed, 1e-6))
        move = mv * scale[:, None]
        move_mag = np.linalg.norm(move, axis=1)

        # --- Reward-free within-lifetime plasticity (Oja) -----------------
        if cfg.hebb_eta > 0:
            dW1 = np.einsum('nh,ni->nhi', h, obs) - (h ** 2)[:, :, None] * self.p_W1
            dW2 = np.einsum('no,nh->noh', mv, h) - (mv ** 2)[:, :, None] * self.p_W2
            self.p_W1 += cfg.hebb_eta * dW1 * a[:, None, None]
            self.p_W2 += cfg.hebb_eta * dW2 * a[:, None, None]
            np.clip(self.p_W1, -cfg.weight_clip, cfg.weight_clip, out=self.p_W1)
            np.clip(self.p_W2, -cfg.weight_clip, cfg.weight_clip, out=self.p_W2)

        # --- Move + physics (only alive matter) ---------------------------
        self.pos = np.clip(pos + move, [0, 0], [cfg.width, cfg.height]).astype(np.float32)
        self.age += a

        nut_un, tox2, cool2 = self.fields.sample(self.pos)
        if cfg.scarce_resources:
            M = self.fields.nutrient_matrix(self.pos)              # [C,S]
            nut2 = (M * self.fields.stock[None, :]).sum(1)         # stock-weighted intake
            # Deplete sources by feeding pressure, then slow regrowth (ecology).
            af = a.astype(np.float32)
            pressure = cfg.nutrient_consumption * (M * af[:, None]).sum(0) * self.fields.stock
            self.fields.stock = np.clip(self.fields.stock - pressure, 0.0, 1.0)
            self.fields.stock += cfg.nutrient_regrowth * (1.0 - self.fields.stock)
        else:
            nut2 = nut_un
        self.E = np.clip(self.E - cfg.basal_cost - cfg.move_cost * move_mag
                         + cfg.nutrient_gain * nut2, 0.0, 1.0).astype(np.float32)
        cooling = cfg.cooling_rate + cfg.cool_zone_cooling * cool2
        self.T = np.clip(self.T + cfg.heat_speed * move_mag + cfg.heat_hazard * tox2
                         - cooling + heat_load, 0.0, 1.5).astype(np.float32)
        overheat = np.maximum(0.0, self.T - cfg.t_crit)
        resting = (move_mag < 0.4 * cfg.max_speed).astype(np.float32)
        repair = cfg.repair_rate * resting * np.where(self.E > cfg.repair_min_energy, 1.0, 0.35)
        self.D = np.clip(self.D + cfg.dmg_toxin * tox2 + cfg.dmg_overheat * overheat
                         - repair, 0.0, cfg.d_max).astype(np.float32)

        # --- Death by inviability (physics decides, no rule authored) -----
        died = a & ((self.E <= 0.0) | (self.T >= cfg.t_fatal) | (self.D >= cfg.d_max))
        cause_star = int((a & (self.E <= 0.0)).sum())
        cause_heat = int((a & (self.T >= cfg.t_fatal)).sum())
        cause_dmg = int((a & (self.D >= cfg.d_max)).sum())
        heat_died = a & (self.T >= cfg.t_fatal)
        heat_deaths_s = int((heat_died & self.senses).sum())
        heat_deaths_y = int((heat_died & ~self.senses).sum())
        if died.any():
            self._lifespan_sum += float(self.age[died].sum())
            self._lifespan_count += int(died.sum())
        self.alive[died] = False
        self.deaths_total += int(died.sum())

        # --- Reproduction as a physical event (enough energy -> split) ----
        births = 0
        can_divide = np.where(self.alive & (self.E > cfg.division_energy)
                              & (self.age >= cfg.min_division_age)
                              & ((self.step_count - self.last_repro) >= cfg.repro_cooldown))[0]
        pop = int(self.alive.sum())
        for pi in can_divide:
            if pop >= cfg.capacity:
                break
            free = np.where(~self.alive)[0]
            if free.size == 0:
                break
            ci = int(free[0])
            avail = float(self.E[pi] - cfg.division_cost)
            if avail <= 0.2:
                continue
            # child genome = parent GENOME (not drifted weights) + mutation
            m = cfg.mutation_std
            genome = (
                np.clip(self.g_W1[pi] + self.rng.normal(0, m, self.g_W1[pi].shape), -cfg.weight_clip, cfg.weight_clip).astype(np.float32),
                np.clip(self.g_W2[pi] + self.rng.normal(0, m, self.g_W2[pi].shape), -cfg.weight_clip, cfg.weight_clip).astype(np.float32),
                (self.b1[pi] + self.rng.normal(0, m, self.b1[pi].shape)).astype(np.float32),
                (self.b2[pi] + self.rng.normal(0, m, self.b2[pi].shape)).astype(np.float32),
            )
            self._spawn(ci, genome, int(self.lineage[pi]), int(self.gen[pi] + 1),
                        senses=bool(self.senses[pi]))
            self.E[pi] = avail * 0.5
            self.E[ci] = avail * 0.5
            self.last_repro[pi] = self.step_count
            self.pos[ci] = np.clip(self.pos[pi] + self.rng.normal(0, 0.5, 2), [0, 0], [cfg.width, cfg.height]).astype(np.float32)
            births += 1
            pop += 1
        self.births_total += births

        al = self.alive
        rec = {
            "step": self.step_count,
            "population": int(al.sum()),
            "mean_E": float(self.E[al].mean()) if al.any() else 0.0,
            "mean_T": float(self.T[al].mean()) if al.any() else 0.0,
            "mean_D": float(self.D[al].mean()) if al.any() else 0.0,
            "births": births,
            "deaths": int(died.sum()),
            "death_starv": cause_star, "death_heat": cause_heat, "death_dmg": cause_dmg,
            "max_gen": int(self.gen[al].max()) if al.any() else 0,
            "lineages": int(np.unique(self.lineage[al]).size) if al.any() else 0,
            "mean_speed": float(move_mag[al].mean()) if al.any() else 0.0,
            "weather": float(self.weather),
            "n_sense": int((al & self.senses).sum()),
            "heat_deaths_s": heat_deaths_s,
            "heat_deaths_y": heat_deaths_y,
        }
        for tag, m in (("S", al & self.senses), ("Y", al & ~self.senses)):
            if m.any():
                rec[f"mean_T_{tag}"] = float(self.T[m].mean())
                rec[f"mean_speed_{tag}"] = float(move_mag[m].mean())
        self.history.append(rec)
        return rec

    def run(self, max_steps: int | None = None):
        self.reset()
        steps = max_steps or self.cfg.max_steps
        for _ in range(steps):
            rec = self.step()
            if rec["population"] == 0:
                break
        return self.history

    # -- causal probe ------------------------------------------------------
    def probe_speed(self, idx: np.ndarray, temps: np.ndarray, cools: np.ndarray | float = 0.0,
                    energy: float = 0.6) -> np.ndarray:
        """Commanded speed of organisms `idx` under clamped inputs — an intervention.

        Food and toxin gradients are silenced, E is held at `energy`, D at 0; the felt
        temperature and local cooling are set to `temps` / `cools` (broadcast to a common
        length K). Uses each organism's current (plastic) weights. Returns [len(idx), K]
        speeds in [0, max_speed].
        """
        idx = np.asarray(idx, dtype=int)
        temps, cools = np.broadcast_arrays(np.asarray(temps, np.float32),
                                           np.asarray(cools, np.float32))
        obs = np.zeros((idx.size, temps.size, N_IN), np.float32)
        obs[..., 4] = cools[None, :]
        obs[..., 5] = energy
        obs[..., 6] = temps[None, :]
        h = np.tanh(np.einsum('nhi,nki->nkh', self.p_W1[idx], obs) + self.b1[idx][:, None, :])
        mv = np.tanh(np.einsum('noh,nkh->nko', self.p_W2[idx], h) + self.b2[idx][:, None, :])
        return np.minimum(np.linalg.norm(mv, axis=-1), self.cfg.max_speed)

    def probe_thermal_reflex(self, idx: np.ndarray, temps: np.ndarray,
                             energy: float = 0.6) -> np.ndarray:
        """Speed vs clamped felt temperature, no cooling present. Falling = slow when hot."""
        return self.probe_speed(idx, temps, 0.0, energy)
