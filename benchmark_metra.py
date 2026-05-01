#!/usr/bin/env python3
"""
benchmark_metra.py — load a METRA checkpoint, roll out skills on Ant-v5, compute & display metrics.

Usage (run from /workspace/METRA_state/):
  python benchmark_metra.py \
      --checkpoint exp/Debug/sd001_1777556593_ant_v5_metra/itr_9000.pkl \
      --n_skills 16 \
      --n_eps 3 \
      --max_steps 200 \
      --output_dir ./benchmark_out

Environment variables (optional):
  TIME_PATH   path to TIME repo containing metrics.py  (default: /workspace/TIME)
  METRA_PATH  METRA source root if not running from it (default: current dir)
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')  # headless — no X display needed

import matplotlib.pyplot as plt
import numpy as np
import torch


# ---------------------------------------------------------------------------
# Path setup — allow running from /workspace/METRA_state/ or any dir
# ---------------------------------------------------------------------------
METRA_PATH = os.environ.get('METRA_PATH', os.path.dirname(os.path.abspath(__file__)))
TIME_PATH  = os.environ.get('TIME_PATH',  '/workspace/TIME')

for p in [METRA_PATH, TIME_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

os.environ.setdefault('MUJOCO_GL', 'egl')

# dowel_wrapper MUST be imported before dowel (and before cloudpickle.load triggers garage imports)
import dowel_wrapper  # noqa: E402  (intentional late import after sys.path setup)
assert dowel_wrapper is not None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_checkpoint(path: str):
    """Load a cloudpickle checkpoint and return the saved dict."""
    import cloudpickle
    with open(path, 'rb') as f:
        return cloudpickle.load(f)


def make_env(seed: int = 0):
    from envs.ant_v5_env import AntV5Env
    return AntV5Env(seed=seed)


def generate_options(dim_option: int, n_skills: int, discrete: bool, unit_length: bool) -> np.ndarray:
    """Return (n_skills, dim_option) array of skills."""
    if discrete:
        # one-hot, cycling if n_skills > dim_option
        idx = np.arange(n_skills) % dim_option
        return np.eye(dim_option)[idx]
    if dim_option == 2:
        # evenly-spaced angles on unit circle — interpretable for 2D
        angles = np.linspace(0, 2 * np.pi, n_skills, endpoint=False)
        opts = np.stack([np.cos(angles), np.sin(angles)], axis=1).astype(np.float32)
    else:
        rng = np.random.default_rng(42)
        opts = rng.standard_normal((n_skills, dim_option)).astype(np.float32)
        if unit_length:
            opts /= np.linalg.norm(opts, axis=1, keepdims=True) + 1e-8
    return opts


def rollout_skill(policy, env, option: np.ndarray, max_steps: int, deterministic: bool = True):
    """
    Roll out one episode with a fixed skill option.

    Returns:
        xs, ys   — list of x positions, y positions (from info['coordinates'])
        total_return  — float
        obs_list      — list of obs arrays (each step's observation)
    """
    obs = env.reset()
    xs, ys, obs_list = [], [], []
    total_return = 0.0

    for _ in range(max_steps):
        concat_obs = np.concatenate([obs, option], axis=0)
        if deterministic:
            action, _ = policy.get_mode_actions(concat_obs[None])  # batch dim
            action = action[0]
        else:
            action, _ = policy.get_action(concat_obs)

        obs, reward, done, info = env.step(action)
        total_return += float(reward)
        obs_list.append(obs.copy())

        coord = info.get('coordinates', np.zeros(2))
        xs.append(float(coord[0]))
        ys.append(float(coord[1]))

        if done:
            break

    return xs, ys, total_return, obs_list


def compute_coverage_score(all_xy, bin_size: float = 1.0) -> float:
    """Number of unique grid bins visited across all skills."""
    bins = set()
    for x, y in all_xy:
        bins.add((int(np.floor(x / bin_size)), int(np.floor(y / bin_size))))
    return float(len(bins))


# ---------------------------------------------------------------------------
# Metrics import (TIME/metrics.py)
# ---------------------------------------------------------------------------

try:
    from metrics import Metrics
    HAS_TIME_METRICS = True
except ImportError:
    HAS_TIME_METRICS = False
    print(f"[warn] Could not import Metrics from {TIME_PATH}. "
          "Coverage plot will use inline implementation.")

    class Metrics:  # minimal inline fallback
        def __init__(self):
            self._locs: dict = {}
            self._returns: list = []

        def save_ant_location(self, algoname, skill, x, y):
            key = (algoname, skill)
            self._locs.setdefault(key, []).append((x, y))

        def save_return(self, algoname, ret):
            self._returns.append(ret)

        def print_return_metrics(self, algoname):
            rets = self._returns
            if not rets:
                print(f"{algoname}: no returns recorded")
                return
            mu, sd = float(np.mean(rets)), float(np.std(rets))
            print(f"{algoname} Return — Mean: {mu:.2f}  Std: {sd:.2f}  over {len(rets)} episodes")

        def plot_ant_explore_metrics(self, algoname, file=None, custom_title=None, legend=True):
            fig, ax = plt.subplots(figsize=(8, 8))
            colors = plt.cm.get_cmap('tab20', max(len(self._locs), 1))
            for i, ((_, skill), locs) in enumerate(self._locs.items()):
                xs, ys = zip(*locs)
                ax.plot(xs, ys, alpha=0.7, color=colors(i), label=f'skill {skill}')
            ax.set_title(custom_title or f'{algoname} coverage')
            ax.set_xlabel('x'); ax.set_ylabel('y')
            if legend:
                ax.legend(fontsize=6, ncol=2, loc='upper right')
            plt.tight_layout()
            if file:
                plt.savefig(file, dpi=150)
                print(f"  saved → {file}")
            plt.close(fig)

        def plot_traj_confusion_matrix(self, encoder, *traj, file=None, custom_title=None):
            observations = list(traj[0]) if len(traj) == 1 else list(traj)
            embs = np.array([encoder(o) for o in observations])
            norms = np.linalg.norm(embs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            norm_embs = embs / norms
            sim = np.clip(norm_embs @ norm_embs.T, -1, 1)
            fig, ax = plt.subplots(figsize=(6, 6))
            im = ax.imshow(sim, cmap='viridis', vmin=-1, vmax=1)
            ax.set_title(custom_title or 'Trajectory Embedding Similarity')
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            if file:
                plt.savefig(file, dpi=150)
                print(f"  saved → {file}")
            plt.close(fig)
            return float(np.mean(sim))

        def save_metrics(self, file='metrics.pkl'):
            import pickle
            with open(file, 'wb') as f:
                pickle.dump({'locs': self._locs, 'returns': self._returns}, f)

        def load_metrics(self, file='metrics.pkl'):
            import pickle
            with open(file, 'rb') as f:
                data = pickle.load(f)
            self._locs    = data.get('locs', {})
            self._returns = data.get('returns', [])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Benchmark METRA checkpoint on Ant-v5')
    p.add_argument('--checkpoint', type=str,
                   default='exp/Debug/sd001_1777556593_ant_v5_metra/itr_9000.pkl',
                   help='Path to itr_*.pkl checkpoint')
    p.add_argument('--n_skills', type=int, default=16,
                   help='Number of distinct skills to evaluate')
    p.add_argument('--n_eps', type=int, default=3,
                   help='Episodes per skill')
    p.add_argument('--max_steps', type=int, default=200,
                   help='Max steps per episode')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--output_dir', type=str, default=None,
                   help='Where to save plots (default: next to checkpoint)')
    p.add_argument('--stochastic', action='store_true',
                   help='Sample actions stochastically (default: deterministic)')
    p.add_argument('--bin_size', type=float, default=1.0,
                   help='Grid bin size for coverage score computation')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- output dir ----
    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), 'benchmark_out')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n=== METRA Benchmark  |  checkpoint: {args.checkpoint}")
    print(f"    output dir: {args.output_dir}\n")

    # ---- load checkpoint ----
    print("[1/5] Loading checkpoint …")
    saved = load_checkpoint(args.checkpoint)
    algo  = saved['algo']
    policy       = algo.option_policy
    traj_encoder = algo.traj_encoder
    dim_option   = algo.dim_option
    discrete     = algo.discrete
    unit_length  = getattr(algo, 'unit_length', True)
    device       = algo.device

    policy.eval()
    traj_encoder.eval()

    print(f"      dim_option={dim_option}  discrete={discrete}  unit_length={unit_length}")
    print(f"      device: {device}")

    # ---- env ----
    print("[2/5] Creating Ant-v5 environment …")
    env = make_env(seed=args.seed)

    # ---- skills ----
    print(f"[3/5] Generating {args.n_skills} skills …")
    options = generate_options(dim_option, args.n_skills, discrete, unit_length)

    # ---- rollouts ----
    print(f"[4/5] Rolling out {args.n_skills} skills × {args.n_eps} episodes "
          f"(max {args.max_steps} steps each) …")
    metrics = Metrics()
    algo_name = 'METRA_ant_v5'

    all_xy              = []
    last_obs_per_skill  = []    # one representative obs per skill for confusion matrix
    skill_locs          = {}    # si -> list of (x, y)  — local copy for scatter plot

    for si, option in enumerate(options):
        skill_locs[si] = []
        skill_rets = []

        ep_xs_all, ep_ys_all = [], []
        ep_last_obs = []

        for ep in range(args.n_eps):
            xs, ys, ret, obs_list = rollout_skill(
                policy, env, option,
                max_steps=args.max_steps,
                deterministic=not args.stochastic,
            )
            skill_rets.append(ret)
            metrics.save_return(algo_name, ret)
            for x, y in zip(xs, ys):
                metrics.save_ant_location(algo_name, si, x, y)
                all_xy.append((x, y))
                skill_locs[si].append((x, y))
            ep_xs_all.extend(xs)
            ep_ys_all.extend(ys)
            if obs_list:
                ep_last_obs.append(obs_list[-1])

        mean_ret = np.mean(skill_rets)
        max_dist = max(np.hypot(x, y) for x, y in zip(ep_xs_all, ep_ys_all)) if ep_xs_all else 0
        print(f"    skill {si:3d}  option=[{', '.join(f'{v:.2f}' for v in option)}]"
              f"  ret={mean_ret:7.1f}  max_dist={max_dist:.2f}")

        if ep_last_obs:
            last_obs_per_skill.append(ep_last_obs[-1])

    # ---- compute coverage score ----
    cov_score = compute_coverage_score(all_xy, bin_size=args.bin_size)
    print(f"\n    Coverage score (unique {args.bin_size}×{args.bin_size} bins): {cov_score:.0f}")

    # ---- display metrics ----
    print("\n[5/5] Computing & saving plots …")

    # Coverage trajectory plot
    cov_file = os.path.join(args.output_dir, 'coverage.png')
    metrics.plot_ant_explore_metrics(
        algo_name,
        file=cov_file,
        custom_title=f'METRA Ant-v5 Coverage  (itr={os.path.basename(args.checkpoint)})',
        legend=(args.n_skills <= 20),
    )

    # Return stats
    print()
    metrics.print_return_metrics(algo_name)

    # Trajectory confusion matrix (encode last obs of each skill)
    if last_obs_per_skill:
        def encoder(obs):
            with torch.no_grad():
                t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
                dist = traj_encoder(t)
                return dist.mean.squeeze(0).cpu().numpy()

        conf_file = os.path.join(args.output_dir, 'confusion_matrix.png')
        mean_sim = metrics.plot_traj_confusion_matrix(
            encoder,
            last_obs_per_skill,
            file=conf_file,
            custom_title='Skill Embedding Cosine Similarity (last obs per skill)',
        )
        print(f"\n    Mean trajectory embedding similarity: {mean_sim:.4f}")
        print("    (lower → skills occupy more distinct embedding regions)")

    # Summary scatter plot: all trajectories coloured by skill
    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.cm.get_cmap('tab20', args.n_skills)
    for si in range(len(options)):
        locs = skill_locs.get(si, [])
        if not locs:
            continue
        xs, ys = zip(*locs)
        ax.scatter(xs, ys, s=2, alpha=0.5, color=cmap(si), label=f'z{si}')
    ax.set_title(f'METRA Ant-v5 — skill trajectories  (n_skills={args.n_skills})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    if args.n_skills <= 20:
        ax.legend(markerscale=4, fontsize=7, ncol=2, loc='upper right')
    scatter_file = os.path.join(args.output_dir, 'scatter.png')
    plt.tight_layout()
    fig.savefig(scatter_file, dpi=150)
    plt.close(fig)
    print(f"  saved → {scatter_file}")

    # Save raw metrics
    pkl_file = os.path.join(args.output_dir, 'metrics.pkl')
    metrics.save_metrics(pkl_file)
    print(f"  saved → {pkl_file}")

    print(f"\nDone. Results in: {args.output_dir}")
    print(f"  coverage.png         — trajectory lines coloured by skill")
    print(f"  scatter.png          — scatter of all visited positions")
    print(f"  confusion_matrix.png — cosine similarity of skill embeddings")
    print(f"  metrics.pkl          — raw metrics (reload with Metrics().load_metrics())")


if __name__ == '__main__':
    main()
