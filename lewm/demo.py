"""Recruiter-grade demo videos.   python -m lewm.demo [--ckpt ...]

Design rules (kept deliberately strict):
  * The claim is on every frame: title bar + labeled panels
    (GOAL / MODEL IMAGINATION / EXECUTION). Ten-second legibility.
  * Honesty about the architecture: LeWM has NO decoder, so the imagination
    panel visualizes imagined latents by nearest-neighbor retrieval over the
    offline dataset, re-rendered at high resolution from the retrieved sim
    pose — and a footnote on the video says so.
  * Metrics in-frame: live distance-to-goal meter with the success threshold
    marked, step counter, success flash. Scores, not vibes.
  * Native 256x256 renders; one validated dark palette; text wears text
    colors, panel identity is carried by color chips and borders.

Outputs:
  media/lewm_hero.gif           one episode, 3 panels + metric strip
  media/lewm_reacher_demo.gif   2x2 episode grid, goal | execution + meters
  media/lewm_collapse.png       the lambda=0 ablation figure (if ckpt given)
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from .data import TrajectorySlices
from .envs import make
from .model import LeWM
from .planner import CEMConfig, CEMPlanner

MEDIA = pathlib.Path("media")

# Validated dark-mode palette (docs: dataviz reference; surface #1a1a19).
SURFACE = (26, 26, 25)
TEXT = (255, 255, 255)
TEXT_2 = (195, 194, 183)
BLUE = (57, 135, 229)     # execution
ORANGE = (217, 89, 38)    # imagination
AQUA = (25, 158, 112)     # goal
GREEN = (0, 131, 0)       # success state

_FONT_DIR = "/usr/share/fonts/truetype/dejavu"


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"{_FONT_DIR}/{name}", size)
    except OSError:
        return ImageFont.load_default()


# ------------------------------------------------- imagination via NN ------

class LatentIndex:
    """Embeddings of (a subsample of) the offline dataset + their sim poses.
    nearest(z) -> qpos of the dataset frame whose embedding is closest —
    the decoder-free way to look at an imagined latent."""

    def __init__(self, model: LeWM, data_root: str, device: str,
                 stride: int = 2, batch: int = 512):
        frames, poses = [], []
        for f in sorted(pathlib.Path(data_root).glob("ep_*.npz")):
            with np.load(f) as z:
                if "qpos" not in z:
                    raise KeyError(f"{f} lacks qpos — recollect data")
                frames.append(z["obs"][::stride])
                poses.append(z["qpos"][::stride])
        obs = np.concatenate(frames)
        self.qpos = np.concatenate(poses)
        if hasattr(model, "act_proj"):                # DinoWM
            a_dim = model.act_proj.in_features
        else:                                          # LeWM
            a_dim = model.action_encoder.net[0].in_features
        embs = []
        with torch.no_grad():
            for i in range(0, len(obs), batch):
                x = torch.as_tensor(obs[i : i + batch], dtype=torch.float32,
                                    device=device).permute(0, 3, 1, 2) / 255.0
                z, _ = model.encode(
                    x.unsqueeze(1),
                    torch.zeros(len(x), 1, a_dim, device=device))
                embs.append(z[:, 0])
        self.z = torch.cat(embs)                       # (N, 192)
        print(f"[demo] latent index: {len(self.z)} frames")

    def nearest(self, z: torch.Tensor) -> np.ndarray:
        i = (self.z - z.unsqueeze(0)).pow(2).sum(-1).argmin().item()
        return self.qpos[i]


# ------------------------------------------------------- episode capture ---

def run_episode(model: LeWM, index: LatentIndex | None, env,
                rng: np.random.Generator, device: str, frameskip: int = 1,
                replan_every: int | None = None) -> dict:
    """One MPC episode, capturing hires frames + imagination + metrics.
    With frameskip fs > 1 the planner emits action BLOCKS (fs raw actions);
    the imagination panel holds each imagined block-latent's NN frame for
    the fs env steps it spans."""
    planner = CEMPlanner(model, CEMConfig())
    fs = frameskip
    budget_blocks = max(1, env.EVAL_BUDGET // fs)
    success_dist = env.SUCCESS_DIST
    if replan_every is None:
        replan_every = 4 if fs == 1 else 1
    t = lambda x: torch.as_tensor(np.array(x), dtype=torch.float32, device=device)
    a_raw = env.action_dim

    env.reset()
    frames = [env.render()]
    blocks = []
    for _ in range(model.history - 1):
        blk = rng.uniform(-0.5, 0.5, size=(fs, a_raw)).astype(np.float32)
        for a in blk:
            f = env.step(a)
        frames.append(f)
        blocks.append(blk.reshape(-1))
    blocks.append(np.zeros(fs * a_raw, dtype=np.float32))
    goal_qpos, goal_img64, goal_tip = env.sample_goal()
    goal_hi = env.render_pose_demo(goal_qpos)

    live, imag, dists = [env.render_demo()], [], []
    start_dist = float(np.linalg.norm(env.target_point - goal_tip))
    done, steps, executed = False, 0, 0
    while executed < budget_blocks and not done:
        n_exec = min(replan_every, budget_blocks - executed)
        ctx_f = t(frames[-model.history:]).permute(0, 3, 1, 2)
        ctx_a = t(np.stack(blocks[-model.history:]))
        plan, imagined = planner.plan(ctx_f, ctx_a,
                                      t(goal_img64).permute(2, 0, 1),
                                      return_rollout=True)
        for k, blk in enumerate(plan[:n_exec].cpu().numpy()):
            im = (env.render_pose_demo(index.nearest(imagined[k]))
                  if index is not None else None)
            for a in blk.reshape(fs, a_raw):
                f = env.step(a)
                live.append(env.render_demo())
                if im is not None:
                    imag.append(im)          # hold across the block's steps
                dists.append(float(np.linalg.norm(env.target_point - goal_tip)))
                steps += 1
                if dists[-1] < success_dist:
                    done = True
                    break
            frames.append(f)
            blocks[-1] = blk.astype(np.float32)
            blocks.append(np.zeros(fs * a_raw, dtype=np.float32))
            executed += 1
            if done:
                break
    return {"goal": goal_hi, "live": live, "imag": imag, "dists": dists,
            "start_dist": start_dist, "success": done, "steps": steps,
            "success_dist": success_dist}


# ------------------------------------------------------------- composing ---

def _label(draw, x, y, text, chip=None):
    if chip:
        draw.rectangle([x, y + 3, x + 10, y + 13], fill=chip)
        x += 16
    draw.text((x, y), text, font=_font(13, bold=True), fill=TEXT_2)


def _panel(canvas, img, x, y, border):
    p = Image.fromarray(img)
    canvas.paste(p, (x, y))
    d = ImageDraw.Draw(canvas)
    d.rectangle([x - 1, y - 1, x + p.width, y + p.height],
                outline=border, width=2)


def _meter(draw, x, y, w, dist, start, thresh, success):
    frac = min(1.0, max(0.0, 1.0 - dist / max(start, 1e-6)))
    draw.rectangle([x, y, x + w, y + 10], outline=TEXT_2, width=1)
    draw.rectangle([x + 1, y + 1, x + 1 + int((w - 2) * frac), y + 9],
                   fill=GREEN if success else BLUE)
    tx = x + int(w * (1 - thresh / max(start, 1e-6)))
    draw.line([tx, y - 3, tx, y + 13], fill=AQUA, width=2)
    draw.text((x + w + 10, y - 2),
              f"dist {dist:.3f} m", font=_font(12), fill=TEXT)


def compose_hero(ep: dict, size: int = 256) -> list[np.ndarray]:
    gap, m = 8, 14
    w = 3 * size + 2 * gap + 2 * m
    title_h, label_h, strip_h, foot_h = 40, 24, 40, 22
    h = title_h + label_h + size + strip_h + foot_h
    out = []
    n = len(ep["live"])
    for i in range(n + 8):                      # +8 hold frames at the end
        j = min(i, n - 1)
        c = Image.new("RGB", (w, h), SURFACE)
        d = ImageDraw.Draw(c)
        d.text((m, 11), "LeWM — decoder-free world model, planning from pixels",
               font=_font(15, bold=True), fill=TEXT)
        d.text((w - m - 62, 13), f"t = {min(j, n - 2) + 1:2d}",
               font=_font(13), fill=TEXT_2)
        xs = [m, m + size + gap, m + 2 * (size + gap)]
        _label(d, xs[0], title_h + 3, "GOAL (image)", AQUA)
        _label(d, xs[1], title_h + 3, "MODEL IMAGINATION", ORANGE)
        _label(d, xs[2], title_h + 3, "EXECUTION (MuJoCo)", BLUE)
        y0 = title_h + label_h
        _panel(c, ep["goal"], xs[0], y0, AQUA)
        if ep["imag"]:
            _panel(c, ep["imag"][min(j, len(ep["imag"]) - 1)], xs[1], y0, ORANGE)
        done_now = ep["success"] and j >= n - 2
        _panel(c, ep["live"][j], xs[2], y0,
               GREEN if done_now else BLUE)
        if done_now:
            d.text((xs[2] + 10, y0 + 10), "SUCCESS",
                   font=_font(16, bold=True), fill=(120, 220, 120))
        di = ep["dists"][min(j, len(ep["dists"]) - 1)] if ep["dists"] else ep["start_dist"]
        _meter(d, m, y0 + size + 14, 2 * size, di, ep["start_dist"],
               ep["success_dist"], done_now)
        d.text((m, h - foot_h + 2),
               "no decoder: imagined latents shown via nearest-neighbor "
               "retrieval from offline data, re-rendered",
               font=_font(11), fill=TEXT_2)
        out.append(np.asarray(c))
    return out


def compose_grid(eps: list[dict], size: int = 256) -> list[np.ndarray]:
    gap, m, label_h, meter_h = 8, 12, 22, 30
    cell_w = 2 * size + gap
    cell_h = label_h + size + meter_h
    w = 2 * cell_w + gap + 2 * m
    h = 2 * cell_h + gap + 2 * m + 30
    n = max(len(e["live"]) for e in eps)
    out = []
    for i in range(n + 8):
        c = Image.new("RGB", (w, h), SURFACE)
        d = ImageDraw.Draw(c)
        d.text((m, 6), "LeWM goal-image planning — 4 episodes",
               font=_font(14, bold=True), fill=TEXT)
        for k, ep in enumerate(eps[:4]):
            j = min(i, len(ep["live"]) - 1)
            x = m + (k % 2) * (cell_w + gap)
            y = 30 + m + (k // 2) * (cell_h + gap)
            _label(d, x, y, "GOAL", AQUA)
            _label(d, x + size + gap, y, "EXECUTION", BLUE)
            done_now = ep["success"] and j >= len(ep["live"]) - 2
            _panel(c, ep["goal"], x, y + label_h, AQUA)
            _panel(c, ep["live"][j], x + size + gap, y + label_h,
                   GREEN if done_now else BLUE)
            di = ep["dists"][min(j, len(ep["dists"]) - 1)] if ep["dists"] else ep["start_dist"]
            _meter(d, x, y + label_h + size + 10, size + 40, di,
                   ep["start_dist"], ep["success_dist"], done_now)
        out.append(np.asarray(c))
    return out


def _save_gif(frames: list[np.ndarray], path: pathlib.Path, fps: int = 10):
    import imageio.v2 as imageio

    MEDIA.mkdir(exist_ok=True)
    imageio.mimsave(path, frames, fps=fps, loop=0)
    print(f"wrote {path}  ({len(frames)} frames)")


# ------------------------------------------------------- ablation figure ---

def collapse_figure(healthy_ckpt: str, collapse_ckpt: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hp = torch.load(healthy_ckpt, map_location="cpu")
    cp = torch.load(collapse_ckpt, map_location="cpu")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2), facecolor="#1a1a19")
    series = [("with SIGReg (λ=0.09)", hp["history"], "#3987e5"),
              ("no SIGReg (λ=0)", cp["history"], "#d95926")]
    for ax, (idx, title, ylog) in zip(
            axes, [(0, "prediction MSE", True), (2, "latent std", False)]):
        for name, hist, color in series:
            ax.plot([h[idx] for h in hist], color=color, lw=2, label=name)
            ax.annotate(name, (len(hist) - 1, hist[-1][idx]),
                        color=color, fontsize=8, xytext=(4, 0),
                        textcoords="offset points", va="center")
        if ylog:
            ax.set_yscale("log")
        ax.set_title(title, color="white", fontsize=11)
        ax.set_xlabel("epoch", color="#c3c2b7", fontsize=9)
        ax.set_facecolor("#1a1a19")
        ax.tick_params(colors="#c3c2b7", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#3a3a38")
        ax.grid(alpha=0.15, color="#c3c2b7")
    axes[0].legend(fontsize=8, facecolor="#1a1a19", labelcolor="#c3c2b7",
                   edgecolor="#3a3a38", loc="upper right")
    fig.suptitle("Collapse ablation: remove SIGReg and prediction 'improves' "
                 "by destroying the representation",
                 color="white", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = MEDIA / "lewm_collapse.png"
    fig.savefig(out, dpi=150, facecolor="#1a1a19",
                bbox_inches="tight", pad_inches=0.25)
    print(f"wrote {out}")


# ------------------------------------------------------------------- main --

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=pathlib.Path, default="data/ckpt/reacher.pt")
    ap.add_argument("--collapse-ckpt", type=pathlib.Path,
                    default="data/ckpt/collapse.pt")
    ap.add_argument("--env", type=str, default=None)
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--episodes", type=int, default=8,
                    help="episodes to run; best success becomes the hero")
    ap.add_argument("--index-stride", type=int, default=2)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    blob = torch.load(args.ckpt, map_location=dev)
    if blob.get("model_type") == "dinowm":
        from .dinowm import DinoWM
        model = DinoWM(action_dim=blob.get("action_dim", 10),
                       history=blob.get("history_len", 3)).to(dev).eval()
    else:
        model = LeWM(history=blob.get("history_len", 3),
                     action_dim=blob.get("action_dim", 2)).to(dev).eval()
    model.load_state_dict(blob["model"])

    env_name = args.env or blob.get("env", "reacher")
    args.data = args.data or f"data/{env_name}"
    fs = blob.get("frameskip", 1)
    index = LatentIndex(model, args.data, dev, stride=args.index_stride)
    env = make(env_name, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    eps = []
    for i in range(args.episodes):
        ep = run_episode(model, index, env, rng, dev, frameskip=fs)
        print(f"  ep {i+1}: {'success' if ep['success'] else 'fail'} "
              f"in {ep['steps']} steps (final {ep['dists'][-1]:.3f} m)")
        eps.append(ep)

    heroes = [e for e in eps if e["success"] and len(e["live"]) >= 8]
    hero = max(heroes or eps, key=lambda e: e["start_dist"])
    stem = args.ckpt.stem
    _save_gif(compose_hero(hero), MEDIA / f"{stem}_hero.gif")
    grid_eps = sorted(eps, key=lambda e: not e["success"])[:4]
    _save_gif(compose_grid(grid_eps), MEDIA / f"{stem}_grid.gif")

    if args.collapse_ckpt.exists():
        import torch as _t
        cb = _t.load(args.collapse_ckpt, map_location="cpu")
        if cb.get("env", "reacher") == env_name:   # never mix envs in one figure
            collapse_figure(str(args.ckpt), str(args.collapse_ckpt))


if __name__ == "__main__":
    main()
