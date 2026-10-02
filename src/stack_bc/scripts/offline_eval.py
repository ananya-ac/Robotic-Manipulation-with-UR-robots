"""Open-loop comparison of trained policies against the demonstrations they're evaluated on.

Runs in the `robodiff` env (no sim needed). For each demo episode, the policy is reset and fed the
DEMO's own observations exactly as a closed-loop rollout would feed its own - the last n_obs_steps
states, queried every n_action_steps ticks (DP: To=2, every 8 ticks; LSTM-GMM: To=1, every tick,
with its recurrent state carried through the episode). Its executed actions are compared with the
demo's recorded actions:

  - accuracy: EE position error (mm), rotation error (deg), gripper width error (mm), and agreement
    of the open/closed decision (the rollout client snaps width to open/closed at the midpoint)
  - smoothness, per episode: step per tick and acceleration RMS of the commanded EE position,
    vs the demo's own actions (plus the jump at chunk boundaries for chunked policies)

This isolates what the policy has learned from closed-loop effects (drift, contact); use
stack_bc.evaluate for the closed-loop number.

  conda activate robodiff
  python ~/ur_ws_jazzy/src/stack_bc/scripts/offline_eval.py --dataset ~/data/stack_bc/overfit.zarr \\
      --ckpt DP=/path/dp.ckpt --ckpt LSTM-GMM=/path/gmm.ckpt [--episodes 0 1] [--plot out.png]
"""

import argparse
import os
import sys

import numpy as np
import torch
import zarr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_server import DEFAULT_DP_ROOT  # noqa: E402

GRIPPER_THRESHOLD = 0.0425     # midpoint of open (0.085) / closed (0.0), as in stack_bc.evaluate


def rot6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:6]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    b2 = a2 - (b1 * a2).sum(-1, keepdims=True) * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    return np.stack([b1, b2, np.cross(b1, b2)], axis=-2)


def replay_episode(policy, info, states, device):
    """Policy's executed actions over one episode, queried as a rollout would."""
    To, Ta = info["n_obs_steps"], info["n_action_steps"]
    policy.reset()
    out, T = [], len(states)
    for t in range(0, T, Ta):
        idx = [max(t - k, 0) for k in range(To - 1, -1, -1)]   # pad by repeating the first state
        obs = torch.from_numpy(states[idx].astype(np.float32))[None].to(device)
        with torch.no_grad():
            a = policy.predict_action({"obs": obs})["action"][0].cpu().numpy()
        out.append(a[:min(Ta, T - t)])
    return np.concatenate(out)


def metrics(pred, demo, Ta, boundaries_from):
    pos = np.linalg.norm(pred[:, :3] - demo[:, :3], axis=1) * 1000
    Rp, Rd = rot6d_to_matrix(pred[:, 3:9]), rot6d_to_matrix(demo[:, 3:9])
    ang = np.degrees(np.arccos(np.clip((np.einsum("tij,tij->t", Rp, Rd) - 1) / 2, -1, 1)))
    width = np.abs(pred[:, 9] - demo[:, 9]) * 1000
    agree = (pred[:, 9] > GRIPPER_THRESHOLD) == (demo[:, 9] > GRIPPER_THRESHOLD)
    step = np.linalg.norm(np.diff(pred[:, :3], axis=0), axis=1) * 1000
    acc = np.linalg.norm(np.diff(pred[:, :3], 2, axis=0), axis=1) / 0.01
    boundary = step[Ta - 1::Ta] if Ta > 1 else np.array([])
    return dict(pos=pos, ang=ang, width=width, agree=agree, step=step, acc=acc, boundary=boundary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ckpt", action="append", required=True, help="NAME=path (repeatable)")
    parser.add_argument("--episodes", type=int, nargs="*", default=None, help="default: all")
    parser.add_argument("--plot", default=None, help="optional PNG: policies vs demo, per episode")
    parser.add_argument("--dp-root", default=DEFAULT_DP_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--inference-steps", type=int, default=16, help="diffusion policies only (DDIM if < 100)")
    args = parser.parse_args()
    sys.path.insert(0, os.path.expanduser(args.dp_root))
    from policy_server import load_policy  # after sys.path: needs diffusion_policy importable

    root = zarr.open(os.path.expanduser(args.dataset), "r")
    states, actions = root["data/state"][:], root["data/action"][:]
    ends = root["meta/episode_ends"][:]
    starts = np.r_[0, ends[:-1]]
    episodes = args.episodes if args.episodes else list(range(len(ends)))

    results = {}
    demo_m = [metrics(actions[starts[e]:ends[e]], actions[starts[e]:ends[e]], 1, 0) for e in episodes]
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        policy, cfg = load_policy(os.path.expanduser(path), args.device)
        if hasattr(policy, "noise_scheduler") and args.inference_steps < policy.noise_scheduler.config.num_train_timesteps:
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            policy.noise_scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
            policy.num_inference_steps = args.inference_steps
        info = dict(n_obs_steps=int(cfg.n_obs_steps), n_action_steps=int(cfg.n_action_steps))
        preds = {e: replay_episode(policy, info, states[starts[e]:ends[e]], args.device) for e in episodes}
        ms = [metrics(preds[e], actions[starts[e]:ends[e]], info["n_action_steps"], 0) for e in episodes]
        results[name] = (preds, ms, info)

    cat = lambda ms, k: np.concatenate([m[k] for m in ms])  # noqa: E731
    print(f"{len(episodes)} episodes, {sum(ends[e] - starts[e] for e in episodes)} steps\n")
    print(f"{'policy':12s} {'pos mm (mean/p95/max)':>24s} {'rot deg (mean/p95)':>19s} {'width mm':>9s} "
          f"{'grip agree':>10s} {'step mm (mean/max)':>19s} {'accel RMS':>10s} {'chunk jump mm':>14s}")
    d = demo_m
    print(f"{'demo':12s} {'-':>24s} {'-':>19s} {'-':>9s} {'-':>10s} "
          f"{cat(d, 'step').mean():8.1f} / {cat(d, 'step').max():6.1f}   {np.sqrt((cat(d, 'acc') ** 2).mean()):7.2f}   {'-':>14s}")
    for name, (_, ms, info) in results.items():
        pos, ang, w, ag = cat(ms, "pos"), cat(ms, "ang"), cat(ms, "width"), cat(ms, "agree")
        st, ac, b = cat(ms, "step"), cat(ms, "acc"), cat(ms, "boundary")
        jump = f"{b.mean():5.1f} / {b.max():5.1f}" if b.size else "  (no chunks)"
        print(f"{name:12s} {pos.mean():7.1f} / {np.percentile(pos, 95):5.1f} / {pos.max():6.1f} "
              f"{ang.mean():8.2f} / {np.percentile(ang, 95):5.2f} {w.mean():9.1f} {ag.mean() * 100:9.1f}% "
              f"{st.mean():8.1f} / {st.max():6.1f}   {np.sqrt((ac ** 2).mean()):7.2f}   {jump:>14s}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        colors = ["tab:orange", "tab:blue", "tab:green", "tab:red"]
        fig, axes = plt.subplots(len(episodes), 4, figsize=(16, 2.6 * len(episodes)), squeeze=False)
        for r, e in enumerate(episodes):
            demo = actions[starts[e]:ends[e]]
            tt = np.arange(len(demo)) * 0.1
            for c, (j, lab) in enumerate([(0, "x (m)"), (1, "y (m)"), (2, "z (m)"), (9, "width (m)")]):
                ax = axes[r, c]
                ax.plot(tt, demo[:, j], "k-", lw=2.4, label="demo")
                for (name, (preds, _, _)), col in zip(results.items(), colors):
                    ax.plot(tt, preds[e][:, j], "-", color=col, lw=1.2, label=name)
                ax.set_ylabel(lab)
                ax.grid(alpha=0.3)
            axes[r, 0].set_title(f"episode {e}", loc="left", fontsize=10)
        axes[0, 0].legend(fontsize=8)
        for ax in axes[-1]:
            ax.set_xlabel("time (s)")
        fig.suptitle("Open-loop replay on demo observations: policies vs demo actions", fontsize=11)
        fig.tight_layout()
        fig.savefig(os.path.expanduser(args.plot), dpi=80)
        print(f"\nplot: {args.plot}")


if __name__ == "__main__":
    main()
