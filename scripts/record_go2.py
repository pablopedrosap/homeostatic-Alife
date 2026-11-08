#!/usr/bin/env python3
"""Side-by-side video: standard vs self-regulating Go2 under the same damage, with live
motor-temperature bars.

The default scenario was fixed BEFORE the HAB-P1 results existed (so it cannot be a
cherry-picked episode): 75% torque loss on the front-left knee, seed 0 of each condition,
episode seed 20000. Statistics over all seeds live in the results report, not in this video.

    python scripts/record_go2.py                         # pre-fixed scenario
    python scripts/record_go2.py --sweep payload --level 6
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from homeostatic_colony.envs.go2_homeostatic import Go2HomeostaticConfig, Go2HomeostaticEnv

PERT_KEY = {"fault": "fault_severity", "payload": "payload_kg", "hot_start": "hot_start"}
LABEL = {"extrinsic": "Standard policy (task reward)", "hybrid": "Self-regulating policy (task + motor homeostasis)"}
LEGS = ["FL", "FR", "RL", "RR"]
INK, INK2, BG = (16, 26, 23), (75, 87, 82), (244, 246, 245)
OK, WARN, CUT = (42, 120, 214), (235, 104, 52), (227, 73, 72)


def _font(size):
    for f in ("/System/Library/Fonts/SFNSMono.ttf", "/System/Library/Fonts/Menlo.ttc",
              "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"):
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            continue
    return ImageFont.load_default()


def hud(frame: np.ndarray, title: str, info: dict, t: float, derate: float) -> np.ndarray:
    h, w, _ = frame.shape
    panel = Image.new("RGB", (w, 150), BG)
    d = ImageDraw.Draw(panel)
    f_title, f = _font(15), _font(12)
    d.text((10, 6), title, fill=INK, font=f_title)
    status = "FELL" if info["fell"] else f"{info['displacement']:5.1f} m"
    d.text((10, 28), f"t = {t:4.1f} s   distance {status}", fill=CUT if info["fell"] else INK2, font=f)
    # 12 bars, grouped by leg: hip / thigh / knee
    x0, y0, bh, bw, gap = 10, 52, 80, 10, 4
    for li, leg in enumerate(LEGS):
        gx = x0 + li * (3 * (bw + gap) + 22)
        for j in range(3):
            T = float(info["temps"][li * 3 + j])
            x = gx + j * (bw + gap)
            d.rectangle([x, y0, x + bw, y0 + bh], outline=(214, 220, 217))
            fill = OK if T < derate else (WARN if T < 1.0 else CUT)
            d.rectangle([x, y0 + bh * (1 - min(T, 1.0)), x + bw, y0 + bh], fill=fill)
        d.text((gx + 4, y0 + bh + 4), leg, fill=INK2, font=f)
        yl = y0 + bh * (1 - derate)
        d.line([gx - 2, yl, gx + 3 * (bw + gap), yl], fill=WARN, width=1)
    lx = x0 + 4 * (3 * (bw + gap) + 22) + 6
    d.text((lx, y0), "motor temperature", fill=INK2, font=f)
    d.text((lx, y0 + 16), "hip | thigh | knee", fill=INK2, font=f)
    d.text((lx, y0 + 36), "— derating starts", fill=WARN, font=f)
    d.text((lx, y0 + 52), "red = cut out", fill=CUT, font=f)
    return np.concatenate([frame, np.asarray(panel)], axis=0)


def rollout(cond: str, seed: int, ep_seed: int, pert: dict, models: Path, every: int = 2):
    from stable_baselines3 import PPO
    model = PPO.load(str(models / cond / f"seed_{seed}" / "model.zip"), device="cpu")
    env = Go2HomeostaticEnv(Go2HomeostaticConfig(reward_mode=cond, seed=ep_seed, **pert))
    obs, info = env.reset(seed=ep_seed)
    frames, fell_frame = [], None
    for i in range(env.cfg.max_steps):
        a, _ = model.predict(obs, deterministic=True)
        obs, r, term, trunc, info = env.step(a)
        if i % every == 0:
            frames.append(hud(env.render(), LABEL[cond], info, (i + 1) * env.dt, env.cfg.t_derate))
        if term:
            fell_frame = frames[-1]
            break
    env.close()
    return frames, fell_frame, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="fault", choices=list(PERT_KEY))
    ap.add_argument("--level", type=float, default=0.75)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-seed", type=int, default=20000)
    ap.add_argument("--models", type=Path, default=ROOT / "results/habp1_go2/models")
    ap.add_argument("--out", type=Path, default=ROOT / "docs/media/go2_fault_side_by_side.mp4")
    a = ap.parse_args()
    pert = {PERT_KEY[a.sweep]: a.level}
    runs = {c: rollout(c, a.seed, a.episode_seed, pert, a.models) for c in ("extrinsic", "hybrid")}
    n = max(len(v[0]) for v in runs.values())
    video = []
    for i in range(n):
        halves = []
        for c in ("extrinsic", "hybrid"):
            frames = runs[c][0]
            halves.append(frames[i] if i < len(frames) else frames[-1])   # freeze after a fall
        video.append(np.concatenate([halves[0], np.full((halves[0].shape[0], 6, 3), 255, np.uint8), halves[1]], axis=1))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(a.out, video, fps=25, codec="libx264", quality=5, macro_block_size=2)
    imageio.imwrite(a.out.with_suffix(".png"), video[min(len(video) - 1, 300)])
    for c, (_, _, info) in runs.items():
        print(f"{c:<10} distance {info['displacement']:.1f} m  fell={info['fell']}  peak T {info['max_temp']:.2f}")
    print(f"→ {a.out}")


if __name__ == "__main__":
    main()
