"""Serve a trained diffusion_policy checkpoint to the ROS-side eval client (stack_bc.evaluate).

Runs in the `robodiff` conda env (torch + diffusion_policy), separate from the ROS venv:

  conda activate robodiff
  python ~/ur_ws_jazzy/src/stack_bc/scripts/policy_server.py \
      --ckpt ~/diffusion_policy/data/outputs/overfit_stack/checkpoints/latest.ckpt [--port 5555]

--dp-root (default ~/diffusion_policy) is put on sys.path so the checkpoint's classes import from
anywhere: running `python <script>` only adds the SCRIPT's directory to sys.path, not the cwd.

Protocol (ZMQ REQ/REP on localhost, pickled dicts - local and trusted only):
  {"cmd": "info"}               -> {"n_obs_steps", "n_action_steps", "horizon", "obs_dim", "action_dim",
                                    "inference_steps", "ckpt"}
  {"cmd": "reset"}              -> {"ok": True}   start of episode: clears recurrent state (LSTM-GMM)
  {"cmd": "act", "obs": (To, obs_dim) float32}
                                -> {"action": (Ta, action_dim) float32, "latency_s": float}
"action" is the policy's executable slice (predictions To-1 .. To-1+Ta, un-normalized), i.e.
exactly what the paper executes before re-planning. Works for both the diffusion policy (To=2, Ta=8)
and the LSTM-GMM baseline (To=1, Ta=1, stateful - hence "reset").
"""

import argparse
import os
import sys
import time

import dill
import hydra
import numpy as np
import torch
import zmq
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)
DEFAULT_DP_ROOT = os.path.expanduser("~/diffusion_policy")


def load_policy(ckpt_path, device, use_ema=True):
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    workspace = hydra.utils.get_class(cfg._target_)(cfg)
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)
    has_ema = getattr(cfg.training, "use_ema", False) and getattr(workspace, "ema_model", None) is not None
    policy = workspace.ema_model if (use_ema and has_ema) else workspace.model
    # Not chained: RobomimicLowdimPolicy.to() (LSTM-GMM) returns None instead of self.
    policy.to(device)
    policy.eval()
    return policy, cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-ema", action="store_true", help="use the raw model instead of the EMA weights")
    parser.add_argument("--dp-root", default=DEFAULT_DP_ROOT, help="diffusion_policy repo checkout")
    parser.add_argument("--inference-steps", type=int, default=16,
                        help="denoising steps per query. <100 swaps in a DDIM sampler over the same trained "
                             "model (paper Sec. 3.4 / Table 7 real-world: 16); 100 = the full DDPM chain "
                             "(~0.4 s/query on an RTX 4070, too slow for 10 Hz control)")
    args = parser.parse_args()
    sys.path.insert(0, os.path.expanduser(args.dp_root))
    args.ckpt = os.path.expanduser(args.ckpt)

    policy, cfg = load_policy(args.ckpt, args.device, use_ema=not args.no_ema)
    is_diffusion = hasattr(policy, "noise_scheduler")
    if is_diffusion:
        train_steps = policy.noise_scheduler.config.num_train_timesteps
        if args.inference_steps < train_steps:
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            policy.noise_scheduler = DDIMScheduler.from_config(policy.noise_scheduler.config)
        policy.num_inference_steps = args.inference_steps
    info = dict(n_obs_steps=int(cfg.n_obs_steps), n_action_steps=int(cfg.n_action_steps),
                horizon=int(cfg.horizon), obs_dim=int(cfg.obs_dim), action_dim=int(cfg.action_dim),
                inference_steps=int(args.inference_steps) if is_diffusion else None,
                ckpt=os.path.abspath(args.ckpt))

    # Warm-up pass (CUDA init/kernels) so the first real request isn't slow.
    with torch.no_grad():
        policy.predict_action({"obs": torch.zeros(1, info["n_obs_steps"], info["obs_dim"], device=args.device)})
    policy.reset()  # recurrent policies (LSTM-GMM) must not start episodes with warm-up state

    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://127.0.0.1:{args.port}")
    print(f"policy server ready on port {args.port}: {info}", flush=True)
    while True:
        req = sock.recv_pyobj()
        if req.get("cmd") == "info":
            sock.send_pyobj(info)
            continue
        if req.get("cmd") == "reset":
            policy.reset()  # start-of-episode: clears recurrent state (no-op for diffusion policies)
            sock.send_pyobj({"ok": True})
            continue
        if req.get("cmd") != "act":
            sock.send_pyobj({"error": f"unknown cmd {req.get('cmd')!r}"})
            continue
        obs = np.asarray(req["obs"], dtype=np.float32)
        if obs.shape != (info["n_obs_steps"], info["obs_dim"]):
            sock.send_pyobj({"error": f"obs shape {obs.shape}, expected {(info['n_obs_steps'], info['obs_dim'])}"})
            continue
        t0 = time.perf_counter()
        with torch.no_grad():
            out = policy.predict_action({"obs": torch.from_numpy(obs)[None].to(args.device)})
        action = out["action"][0].cpu().numpy().astype(np.float32)
        sock.send_pyobj({"action": action, "latency_s": time.perf_counter() - t0})


if __name__ == "__main__":
    main()
