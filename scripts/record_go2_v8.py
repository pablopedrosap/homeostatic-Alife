#!/usr/bin/env python3
"""Side-by-side video for HAB-P8: today's practice (standard gait + reactive thermal rule) vs v8
(heat-aware gait + anticipatory governor) on the courier route with three 6-kg carries.

Default = robot 0, first evaluation episode (seed 80000): the same "first robot, first episode"
convention as the HAB-P1 video, not picked from results. One episode is an illustration only.

    python scripts/record_go2_v8.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import scripts.run_go2_v8 as R
from scripts.record_go2 import CUT, INK, INK2, BG, OK, WARN, LEGS, _font
from homeostatic_colony.control.anticipatory import AnticipatoryGovernor
from homeostatic_colony.control.central_governor import BodySchema, sense_temperature, wear_rate

TITLES = {("standard", "reactive"): "Today: standard gait + slow-when-hot rule",
          ("heat_aware", "anticipatory"): "v8: heat-aware gait + planning governor"}


def panel(frame, title, t, kg, cmd, dist, wear, temps, fell):
    h, w, _ = frame.shape
    p = Image.new("RGB", (w, 168), BG)
    d = ImageDraw.Draw(p)
    d.text((10, 6), title, fill=INK, font=_font(14))
    load = "CARRYING 6 kg" if kg > 0 else "empty"
    d.text((10, 27), f"t = {t:5.1f} s   {load:<14} speed cmd {cmd:.2f} m/s", fill=WARN if kg > 0 else INK2, font=_font(12))
    d.text((10, 44), ("FELL" if fell else f"walked {dist:5.1f} m") + f"   motor wear {wear/1000:6.1f} k", fill=CUT if fell else INK2, font=_font(12))
    x0, y0, bh, bw, gap = 10, 66, 76, 10, 4
    for li, leg in enumerate(LEGS):
        gx = x0 + li * (3 * (bw + gap) + 22)
        for j in range(3):
            T = float(temps[li * 3 + j])
            x = gx + j * (bw + gap)
            d.rectangle([x, y0, x + bw, y0 + bh], outline=(214, 220, 217))
            d.rectangle([x, y0 + bh * (1 - min(T, 1.0)), x + bw, y0 + bh], fill=OK if T < 0.7 else (WARN if T < 1.0 else CUT))
        d.text((gx + 4, y0 + bh + 4), leg, fill=INK2, font=_font(11))
        yl = y0 + bh * 0.3
        d.line([gx - 2, yl, gx + 3 * (bw + gap), yl], fill=WARN, width=1)
    lx = x0 + 4 * (3 * (bw + gap) + 22) + 6
    d.text((lx, y0), "motor temperature", fill=INK2, font=_font(11))
    d.text((lx, y0 + 15), "hip | thigh | knee", fill=INK2, font=_font(11))
    d.text((lx, y0 + 32), "— losing strength", fill=WARN, font=_font(11))
    return np.concatenate([frame, np.asarray(p)], axis=0)


def rollout(gait, pacing, seed, ep_seed, every):
    model = R._policy(gait, seed)
    sched = R.MISSIONS["bursts"]
    env = R._env(ep_seed, cmd_vx=1.0, max_steps=int(R.MISSION_S / 0.02), governor=(pacing == "reactive"))
    gov = None
    if pacing == "anticipatory":
        sd = json.loads((R.OUT / "schema" / f"{gait}_seed_{seed}.json").read_text())
        gov = AnticipatoryGovernor(BodySchema.from_json(sd["schema"]), R.MISSION_S, sched,
                                   {float(k): v for k, v in sd["load_scales"].items()}, dt=env.dt)
    obs, _ = env.reset(seed=ep_seed)
    prev = env.data.qpos[0:2].copy()
    dist = wear = 0.0
    q = np.zeros(12)
    cur = None
    frames, info = [], {"temps": env.temp, "fell": False, "cmd": 1.0}
    for t in range(env.cfg.max_steps):
        tt = t * env.dt
        kg = next(ph[2] for i, ph in enumerate(sched) if ph[0] <= tt < ph[1] or i == len(sched) - 1)
        if kg != cur:
            cur = kg
            R.set_payload(env, kg)
        if gov is not None:
            env.cmd = gov.update(sense_temperature(env.temp), q)
            obs = env._obs()
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        q = info["torque_sq"]
        xy = env.data.qpos[0:2]
        dist += float(np.linalg.norm(xy - prev))
        prev = xy.copy()
        wear += float(wear_rate(info["temps"]).sum()) * env.dt
        if t % every == 0 or term:
            frames.append(panel(env.render(), TITLES[(gait, pacing)], tt, kg, info["cmd"], dist, wear,
                                info["temps"], info["fell"]))
        if term or trunc:
            break
    env.close()
    return frames, dist, wear, info["fell"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-seed", type=int, default=R.EP_SEED0)
    ap.add_argument("--every", type=int, default=8, help="render every Nth control step (8 → 4× speed at 25 fps... ≈ 45 s)")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/media/go2_v8_courier.mp4")
    a = ap.parse_args()
    left = rollout("standard", "reactive", a.seed, a.episode_seed, a.every)
    right = rollout("heat_aware", "anticipatory", a.seed, a.episode_seed, a.every)
    n = max(len(left[0]), len(right[0]))
    video = []
    for i in range(n):
        L = left[0][min(i, len(left[0]) - 1)]
        Rf = right[0][min(i, len(right[0]) - 1)]
        video.append(np.concatenate([L, np.full((L.shape[0], 6, 3), 255, np.uint8), Rf], axis=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, video, fps=25, codec="libx264", quality=5, macro_block_size=2)
    imageio.imwrite(a.out.with_suffix(".png"), video[min(len(video) - 1, int(48 / (a.every * 0.02)))])
    for name, (fr, dist, wear, fell) in (("today's practice", left), ("v8", right)):
        print(f"{name:<17} walked {dist:6.1f} m  wear {wear/1000:6.1f}k  wear/m {wear/max(dist,1):6.0f}  fell={fell}")
    print(f"→ {a.out}  ({len(video)} frames, {len(video)/25:.0f} s)")


if __name__ == "__main__":
    main()
