"""Go2 "body budget" experiments on GPU (MuJoCo MJX + Brax PPO), for Kaggle / Colab.

Same physics as the CPU environment in homeostatic_colony/envs/go2_homeostatic.py, rewritten in JAX
so thousands of robots run in parallel on one GPU:

  * Unitree Go2 (MuJoCo Menagerie, pinned commit), 250 Hz physics, 50 Hz policy, PD control
    (kp 50, kd 0.5) computed here so torque limits can differ per robot.
  * 12 motors heat with (τ/τ_peak)² and cool with time constant τ = 10 s; above T = 0.7 the driver
    derates torque linearly, at T = 1.0 the motor cuts out.
  * Interoception: motor temperatures + effort (τ/τ_peak)², with a short memory (last 3 steps).
  * Development under variability: speed commands 0.4–1.2 m/s, hot starts, loads 0–8 kg that switch
    mid-walk (modelled as a downward force on the trunk), a random weak motor.
  * Reward: task (velocity tracking etc.) + w · homeostatic drive (per-step cost of motor
    temperature above a setpoint). w = 0 is the standard robot.
  * Wear: Montsinger's rule, aging rate 2^(7.5·T) per motor; a "lifetime" carries permanent strength
    loss between missions.

Usage (see kaggle/README.md and the notebook):
    xml = fetch_go2()
    params, inference, metrics = train(xml, homeo_weight=0.2, num_timesteps=30_000_000)
    print(mission(xml, params, inference, payload_kg=3.0))
"""

from __future__ import annotations

import functools
import subprocess
import time
from pathlib import Path

import jax
import jax.numpy as jp
import mujoco
import numpy as np
from brax.envs.base import Env, State
from mujoco import mjx

MENAGERIE_URL = "https://github.com/google-deepmind/mujoco_menagerie.git"
MENAGERIE_COMMIT = "98a70c569c4c1b6da467a20012f766ee1f8dc521"
W_AGE = 32_000.0


def fetch_go2(dest: str = "mujoco_menagerie") -> str:
    """Download the Unitree Go2 model (sparse checkout, pinned). Returns scene_mjx.xml path."""
    d = Path(dest)
    xml = d / "unitree_go2" / "scene_mjx.xml"
    if not xml.exists():
        subprocess.run(["git", "clone", "--filter=blob:none", "--no-checkout", MENAGERIE_URL, str(d)], check=True)
        subprocess.run(["git", "sparse-checkout", "set", "unitree_go2"], cwd=d, check=True)
        subprocess.run(["git", "checkout", MENAGERIE_COMMIT], cwd=d, check=True)
    return str(xml)


def load_torque_model(xml_path: str, sim_dt: float = 0.004):
    """Load the Go2 and turn its position actuators into pure torque motors (force = ctrl)."""
    m = mujoco.MjModel.from_xml_path(xml_path)
    m.opt.timestep = sim_dt
    peak = m.actuator_forcerange[:, 1].copy()
    m.actuator_gainprm[:, :] = 0.0
    m.actuator_gainprm[:, 0] = 1.0
    m.actuator_biasprm[:, :] = 0.0
    m.actuator_biastype[:] = mujoco.mjtBias.mjBIAS_NONE
    m.actuator_ctrlrange[:, 0] = -peak
    m.actuator_ctrlrange[:, 1] = peak
    m.actuator_ctrllimited[:] = 1
    return m, peak


class Go2BodyBudget(Env):
    def __init__(self, xml_path: str, homeo_weight: float = 0.0, t_setpoint: float = 0.6,
                 cmd_range=(0.4, 1.2), t_init_high: float = 0.8, payload_max: float = 8.0,
                 payload_switch_rate: float = 0.1, fault_max: float = 0.5, history: int = 3,
                 kp: float = 50.0, kd: float = 0.5, n_sub: int = 5, sim_dt: float = 0.004,
                 tau_thermal: float = 10.0, heat_gain: float = 0.57, t_derate: float = 0.7,
                 t_cutoff: float = 1.0, action_scale: float = 0.25):
        self.mj_model, peak = load_torque_model(xml_path, sim_dt)
        self.sys = mjx.put_model(self.mj_model)
        self.peak = jp.asarray(peak)
        self.home_qpos = jp.asarray(self.mj_model.key_qpos[0])
        self.home_q = jp.asarray(self.mj_model.key_ctrl[0])       # joint targets of the home pose
        self.base = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
        self.base_mass = float(self.mj_model.body_mass.sum())
        self.w, self.sp = float(homeo_weight), float(t_setpoint)
        self.cmd_range, self.t_init_high = cmd_range, float(t_init_high)
        self.payload_max, self.switch_rate, self.fault_max = float(payload_max), float(payload_switch_rate), float(fault_max)
        self.history, self.kp, self.kd, self.n_sub = history, kp, kd, n_sub
        self.dt = sim_dt * n_sub
        self.tau, self.gain = tau_thermal, heat_gain
        self.t_derate, self.t_cutoff, self.action_scale = t_derate, t_cutoff, action_scale
        self.per_obs = 70

    # -- Brax Env interface ------------------------------------------------------------------
    @property
    def observation_size(self) -> int:
        return self.per_obs * self.history

    @property
    def action_size(self) -> int:
        return 12

    @property
    def backend(self) -> str:
        return "mjx"

    def reset(self, rng: jax.Array) -> State:
        k = jax.random.split(rng, 8)
        cmd = jax.random.uniform(k[0], (), minval=self.cmd_range[0], maxval=self.cmd_range[1])
        temps = jax.random.uniform(k[1], (12,), maxval=self.t_init_high)
        payload = jax.random.uniform(k[2], (), maxval=self.payload_max)
        weak = jax.random.randint(k[3], (), 0, 12)
        strength = jp.ones(12).at[weak].set(1.0 - jax.random.uniform(k[4], (), maxval=self.fault_max))
        return self.reset_with(k[5], cmd, temps, payload, strength, switch=True)

    def reset_with(self, rng, cmd, temps, payload, strength, switch=False) -> State:
        """Reset with explicit conditions (used for missions/lifetimes)."""
        k1, k2, k3 = jax.random.split(rng, 3)
        qpos = self.home_qpos.at[7:].add(jax.random.uniform(k1, (12,), minval=-0.05, maxval=0.05))
        qvel = jax.random.uniform(k2, (self.sys.nv,), minval=-0.05, maxval=0.05)
        data = mjx.make_data(self.sys).replace(qpos=qpos, qvel=qvel)
        data = mjx.forward(self.sys, data)
        info = {"rng": k3, "cmd": jp.asarray(cmd, jp.float32), "temps": jp.asarray(temps, jp.float32),
                "payload": jp.asarray(payload, jp.float32), "strength": jp.asarray(strength, jp.float32),
                "switch": jp.asarray(switch, jp.float32), "last_action": jp.zeros(12), "last_sq": jp.zeros(12),
                "wear": jp.zeros(12), "path": jp.zeros(()), "prev_xy": qpos[0:2]}
        single = self._single_obs(data, info)
        info["hist"] = jp.tile(single, (self.history, 1))
        obs = info["hist"].reshape(-1)
        metrics = {"forward_vel": jp.zeros(()), "hottest_motor": jp.max(info["temps"]), "r_homeo": jp.zeros(()),
                   "payload_kg": info["payload"]}
        return State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    def step(self, state: State, action: jax.Array) -> State:
        info = dict(state.info)
        a = jp.clip(action, -1.0, 1.0)
        target = self.home_q + self.action_scale * a
        derate = jp.clip((self.t_cutoff - info["temps"]) / (self.t_cutoff - self.t_derate), 0.0, 1.0)
        limit = self.peak * derate * info["strength"] + 1e-6
        load = jp.zeros((self.sys.nbody, 6)).at[self.base, 2].set(-9.81 * info["payload"])

        def substep(carry, _):
            data, sq = carry
            q, qd = data.qpos[7:], data.qvel[6:]
            tau = jp.clip(self.kp * (target - q) - self.kd * qd, -limit, limit)
            data = mjx.step(self.sys, data.replace(ctrl=tau, xfrc_applied=load))
            return (data, sq + (tau / self.peak) ** 2), tau

        (data, sq), taus = jax.lax.scan(substep, (state.pipeline_state, jp.zeros(12)), None, length=self.n_sub)
        sq = sq / self.n_sub
        temps = jp.maximum(0.0, info["temps"] + self.dt * (self.gain * sq - info["temps"] / self.tau))

        # loads that change mid-walk (development under variability)
        rng, k1, k2 = jax.random.split(info["rng"], 3)
        switch = info["switch"] * (jax.random.uniform(k1) < self.switch_rate * self.dt)
        payload = jp.where(switch > 0, jax.random.uniform(k2, (), maxval=self.payload_max), info["payload"])

        v, w, g = self._base_frame(data)
        fell = (data.qpos[2] < 0.15) | (g[2] > -0.5)
        tau_last = taus[-1]
        r_task = (jp.exp(-((v[0] - info["cmd"]) ** 2) / 0.25) + 0.5 * jp.exp(-(v[1] ** 2 + w[2] ** 2) / 0.25)
                  - 2.0 * v[2] ** 2 - 0.5 * (g[0] ** 2 + g[1] ** 2) - 2e-5 * jp.sum(tau_last ** 2)
                  - 0.01 * jp.sum((a - info["last_action"]) ** 2) + 0.2)
        excess = jp.maximum(0.0, temps - self.sp) / (1.0 - self.sp)
        r_homeo = -jp.sum(excess)
        reward = r_task + self.w * r_homeo

        xy = data.qpos[0:2]
        info.update(rng=rng, temps=temps, payload=payload, last_action=a, last_sq=sq,
                    wear=info["wear"] + jp.power(2.0, 7.5 * temps) * self.dt,
                    path=info["path"] + jp.linalg.norm(xy - info["prev_xy"]), prev_xy=xy)
        single = self._single_obs(data, info)
        info["hist"] = jp.concatenate([info["hist"][1:], single[None]], axis=0)
        obs = info["hist"].reshape(-1)
        bad = ~jp.all(jp.isfinite(obs))
        obs = jp.where(bad, 0.0, obs)
        done = jp.where(fell | bad, 1.0, 0.0)
        metrics = dict(state.metrics)          # keep keys that Brax's wrappers added (e.g. "reward")
        metrics.update(forward_vel=v[0], hottest_motor=jp.max(temps), r_homeo=r_homeo, payload_kg=payload)
        return state.replace(pipeline_state=data, obs=obs, reward=jp.where(bad, 0.0, reward), done=done,
                             metrics=metrics, info=info)

    # -- helpers ------------------------------------------------------------------------------
    def _base_frame(self, data):
        R = data.xmat[self.base]
        v = R.T @ data.qvel[0:3]
        w = data.qvel[3:6]
        g = R.T @ jp.array([0.0, 0.0, -1.0])
        return v, w, g

    def _single_obs(self, data, info):
        v, w, g = self._base_frame(data)
        return jp.concatenate([v * 2.0, w * 0.25, g, info["cmd"][None], data.qpos[7:] - self.home_qpos[7:],
                               data.qvel[6:] * 0.05, info["last_action"], info["temps"], info["last_sq"]])


# ---------------------------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------------------------

def train(xml_path: str, homeo_weight: float = 0.0, t_setpoint: float = 0.6, num_timesteps: int = 30_000_000,
          num_envs: int = 4096, seed: int = 0, episode_length: int = 1000, progress: bool = True, **ppo_overrides):
    """Train one robot with Brax PPO. Returns (params, make_inference_fn, final_metrics)."""
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo

    env = Go2BodyBudget(xml_path, homeo_weight=homeo_weight, t_setpoint=t_setpoint)
    t0 = time.time()
    log = []

    def progress_fn(step, metrics):
        # Brax reports metrics summed over each evaluation episode: divide by its length.
        length = max(float(metrics.get("eval/avg_episode_length", 1.0)), 1.0)
        reward = float(metrics.get("eval/episode_reward", np.nan))
        hottest = float(metrics.get("eval/episode_hottest_motor", np.nan)) / length
        speed = float(metrics.get("eval/episode_forward_vel", np.nan)) / length
        log.append({"step": int(step), "episode_reward": reward, "episode_length": length,
                    "mean_hottest_motor": hottest, "mean_forward_speed": speed, "seconds": time.time() - t0})
        if progress:
            print(f"  [{time.time() - t0:6.0f}s] steps {step:>11,}  reward {reward:8.1f}  episode {length:5.0f} steps  "
                  f"speed {speed:5.2f} m/s  hottest motor {hottest:.2f}", flush=True)

    kw = dict(num_timesteps=num_timesteps, num_evals=6, reward_scaling=1.0, episode_length=episode_length,
              normalize_observations=True, action_repeat=1, unroll_length=20, num_minibatches=32,
              num_updates_per_batch=4, discounting=0.99, learning_rate=3e-4, entropy_cost=1e-2,
              num_envs=num_envs, batch_size=256, seed=seed, clipping_epsilon=0.2,
              network_factory=functools.partial(ppo_networks.make_ppo_networks,
                                                policy_hidden_layer_sizes=(128, 128, 128, 128)))
    kw.update(ppo_overrides)
    make_inference_fn, params, metrics = ppo.train(environment=env, progress_fn=progress_fn, **kw)
    return params, make_inference_fn, {"log": log, "seconds": time.time() - t0}


# ---------------------------------------------------------------------------------------------
# Evaluation: missions and lifetimes
# ---------------------------------------------------------------------------------------------

@functools.lru_cache(maxsize=4)
def _eval_env(xml_path: str):
    env = Go2BodyBudget(xml_path, homeo_weight=0.0, payload_switch_rate=0.0)
    return env, jax.jit(env.reset_with, static_argnames=("switch",)), jax.jit(env.step)


@functools.lru_cache(maxsize=32)
def _rollout_fn(xml_path: str, make_inference_fn, n_steps: int):
    """A whole mission compiled into one GPU call. After a fall the state is frozen (exact)."""
    env, _, _ = _eval_env(xml_path)

    def run(params, state, rng):
        policy = make_inference_fn(params, deterministic=True)

        def body(carry, _):
            state, rng, alive, frozen, steps = carry
            rng, k = jax.random.split(rng)
            action, _ = policy(state.obs, k)
            nxt = env.step(state, action)
            nxt = jax.tree_util.tree_map(lambda a, b: jp.where(alive > 0, a, b), nxt, state)
            frozen = frozen + alive * (nxt.metrics["forward_vel"] < 0.1)
            steps = steps + alive
            alive = alive * (1.0 - nxt.done)
            return (nxt, rng, alive, frozen, steps), None

        init = (state, rng, jp.float32(1.0), jp.float32(0.0), jp.float32(0.0))
        (state, _, alive, frozen, steps), _ = jax.lax.scan(body, init, None, length=n_steps)
        return state, alive, frozen, steps

    return jax.jit(run)


def mission(xml_path, params, make_inference_fn, payload_kg=0.0, seconds=180.0, seed=0, strength=None,
            record_every: int = 0):
    """One mission at 1 m/s. Returns metrics (+ per-motor wear, and recorded states if record_every > 0)."""
    env, reset_fn, step_fn = _eval_env(xml_path)
    strength = jp.ones(12) if strength is None else jp.asarray(strength, jp.float32)
    rng = jax.random.PRNGKey(seed)
    state = reset_fn(rng, jp.float32(1.0), jax.random.uniform(rng, (12,), maxval=0.3),
                     jp.float32(payload_kg), strength, switch=False)
    n = int(seconds / env.dt)
    frames = []
    if record_every:                      # step by step, keeping frames for the video
        policy = jax.jit(make_inference_fn(params, deterministic=True))
        frozen = 0
        for t in range(n):
            rng, k = jax.random.split(rng)
            action, _ = policy(state.obs, k)
            state = step_fn(state, action)
            frozen += int(float(state.metrics["forward_vel"]) < 0.1)
            if t % record_every == 0:
                frames.append(state.pipeline_state)
            if float(state.done) > 0:
                break
        steps, fell = t + 1, float(state.done) > 0
    else:                                 # whole mission in one compiled call
        state, alive, frozen, steps = _rollout_fn(xml_path, make_inference_fn, n)(params, state, rng)
        frozen, steps, fell = float(frozen), int(steps), float(alive) == 0.0
    info = state.info
    out = {"path_m": float(info["path"]), "wear": float(jp.sum(info["wear"])),
           "wear_per_m": float(jp.sum(info["wear"])) / max(float(info["path"]), 1.0),
           "fell": bool(fell), "frozen_frac": frozen / max(steps, 1), "seconds": steps * env.dt,
           "wear_by_motor": np.asarray(info["wear"])}
    if record_every:
        out["frames"] = frames
    return out


def lifetime(xml_path, params, make_inference_fn, loads=(3.0, 6.0, 0.0, 6.0), seconds=60.0, seed=0):
    """Missions in sequence; heat damage permanently weakens motors between missions."""
    strength = np.ones(12)
    total, wear, falls = 0.0, 0.0, 0
    for i, kg in enumerate(loads):
        m = mission(xml_path, params, make_inference_fn, kg, seconds, seed + i, strength)
        total += m["path_m"]
        wear += m["wear"]
        falls += int(m["fell"])
        strength = strength * np.exp(-m["wear_by_motor"] / W_AGE)
    return {"fitness": total, "wear": wear, "falls": falls, "final_strength": float(strength.mean())}


def render(xml_path, frames, width=480, height=360):
    """Render recorded mjx states to RGB frames (CPU renderer; needs MUJOCO_GL=egl on servers)."""
    env, _, _ = _eval_env(xml_path)
    m = env.mj_model
    d = mujoco.MjData(m)
    r = mujoco.Renderer(m, height, width)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation, cam.azimuth = 1.6, -18, 140
    out = []
    for f in frames:
        d.qpos[:] = np.asarray(f.qpos)
        d.qvel[:] = np.asarray(f.qvel)
        mujoco.mj_forward(m, d)
        cam.lookat[:] = d.xpos[env.base]
        r.update_scene(d, cam)
        out.append(r.render())
    r.close()
    return out


# ---------------------------------------------------------------------------------------------
# Experiments run by the notebook
# ---------------------------------------------------------------------------------------------

def save_params(params, path):
    import pickle
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(params), f)


def load_params(path):
    import pickle
    with open(path, "rb") as f:
        return pickle.load(f)


def compare_drives(xml_path, weights=(0.0, 0.2), seeds=(0, 1), num_timesteps=50_000_000, payloads=(0.0, 3.0, 6.0),
                   mission_s=180.0, episodes=3, out_dir="results", **train_kw):
    """Part 1: train a standard robot (w = 0) and self-regulating robots (w > 0), then send each on
    3-minute missions carrying each payload. Returns a list of result rows (also saved as CSV)."""
    import csv
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, models = [], {}
    for w in weights:
        for seed in seeds:
            name = f"w{w:g}_seed{seed}"
            print(f"\n=== training {name} ===", flush=True)
            params, make_inf, info = train(xml_path, homeo_weight=w, num_timesteps=num_timesteps, seed=seed, **train_kw)
            save_params(params, out / f"{name}.pkl")
            models[name] = (params, make_inf)
            for kg in payloads:
                for e in range(episodes):
                    m = mission(xml_path, params, make_inf, kg, mission_s, seed=10_000 + e)
                    rows.append({"w": w, "seed": seed, "payload": kg, "episode": e,
                                 **{k: v for k, v in m.items() if k != "wear_by_motor"}})
                    print(f"  mission {kg:g} kg #{e}: path {m['path_m']:.1f} m  wear/m {m['wear_per_m']:.0f}  "
                          f"fell {m['fell']}  frozen {m['frozen_frac']:.0%}", flush=True)
    with open(out / "compare_drives.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    return rows, models


def summarize(rows):
    """Mean path and wear per metre for each drive strength × payload."""
    table = {}
    for r in rows:
        table.setdefault((r["w"], r["payload"]), []).append(r)
    print(f"{'drive w':>8} {'payload':>8} {'path m':>8} {'wear/m':>8} {'falls':>6} {'frozen':>7}")
    for (w, kg), rs in sorted(table.items()):
        print(f"{w:8g} {kg:8g} {np.mean([r['path_m'] for r in rs]):8.1f} {np.mean([r['wear_per_m'] for r in rs]):8.0f} "
              f"{np.mean([r['fell'] for r in rs]):6.0%} {np.mean([r['frozen_frac'] for r in rs]):7.0%}")
    return table


def evolve(xml_path, pop=6, generations=4, num_timesteps=10_000_000, lifetimes=2, out_dir="results", seed=2026,
           mission_s=60.0, **train_kw):
    """Part 2: evolution of the drive (Baldwinian, like genes: instincts are inherited, skills are not).
    Two arms: 'evolved' (drive strength w and setpoint inherited + mutated) and 'control' (w = 0).
    Each generation every individual learns to walk from scratch with its inherited drive, then lives
    lifetimes of 4 one-minute missions with permanent motor aging. The top half of each arm survives
    and reproduces (mutated copies replace the bottom half)."""
    import json
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    arms = {"evolved": [{"w": float(w), "sp": float(sp)} for w, sp in
                        zip(np.geomspace(0.02, 0.5, pop), rng.permutation(np.linspace(0.5, 0.8, pop)))],
            "control": [{"w": 0.0, "sp": 0.6} for _ in range(pop)]}
    history = []
    for g in range(generations):
        loads = [tuple(float(x) for x in np.random.default_rng(seed + 100 * g + l).choice([0.0, 3.0, 6.0], 4))
                 for l in range(lifetimes)]
        for arm, members in arms.items():
            for i, gene in enumerate(members):
                print(f"\n=== generation {g} {arm} #{i}  w={gene['w']:.3f} setpoint={gene['sp']:.2f} ===", flush=True)
                params, make_inf, _ = train(xml_path, homeo_weight=gene["w"], t_setpoint=gene["sp"],
                                            num_timesteps=num_timesteps, seed=1000 * g + i, progress=False, **train_kw)
                lives = [lifetime(xml_path, params, make_inf, loads=ls, seconds=mission_s, seed=500 * g + 10 * l)
                         for l, ls in enumerate(loads)]
                gene["fitness"] = float(np.mean([L["fitness"] for L in lives]))
                gene["final_strength"] = float(np.mean([L["final_strength"] for L in lives]))
                gene["falls"] = float(np.mean([L["falls"] for L in lives]))
                history.append({"generation": g, "arm": arm, **gene})
                print(f"    lifetime distance {gene['fitness']:.1f} m   motor strength left {gene['final_strength']:.2f}", flush=True)
            # selection: top half survives and reproduces; bottom half replaced by mutated copies
            ranked = sorted(members, key=lambda m: -m["fitness"])
            survivors = ranked[: pop // 2]
            children = []
            for parent in survivors[: pop - len(survivors)]:
                child = {"w": parent["w"], "sp": parent["sp"]}
                if arm == "evolved":
                    child["w"] = float(np.clip(parent["w"] * rng.choice([0.67, 1.5]), 0.0, 3.0))
                    child["sp"] = float(np.clip(parent["sp"] + rng.choice([-0.05, 0.05]), 0.4, 0.9))
                children.append(child)
            arms[arm] = [{"w": m["w"], "sp": m["sp"]} for m in survivors] + children
            print(f"--- generation {g} {arm}: median lifetime distance "
                  f"{np.median([m['fitness'] for m in members]):.1f} m; drives now {[round(m['w'], 3) for m in arms[arm]]}", flush=True)
        (out / "evolution_history.json").write_text(json.dumps(history, indent=1))
    return history
