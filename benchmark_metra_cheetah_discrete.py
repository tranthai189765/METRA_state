#!/usr/bin/env python3
"""
benchmark_metra_cheetah_discrete.py — METRA dmc_cheetah_state discrete-skill benchmark

Loads the latest (or specified) option_policy checkpoint from a METRA experiment
directory and rolls out all discrete skills.

Metrics (matching TIME's metrics.py conventions):
  1. X-position KDE density per skill  →  skill_x_density.png
  2. Mean-obs cosine-distance matrix   →  cosine_distance_matrix.png
  3. Step-by-step x positions          →  x_locations_metra_cheetah_discrete.csv

Usage:
    python benchmark_metra_cheetah_discrete.py \\
        --exp_dir /workspace/METRA_state/exp/Debug/sd001_1777660191_dmc_cheetah_state_metra \\
        --n_episodes 1 \\
        --episode_steps 1000 \\
        --cuda_id 0 \\
        --out_dir ./benchmark_results_metra_cheetah_discrete
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import gaussian_kde
from sklearn.metrics.pairwise import cosine_distances

sys.path.insert(0, str(Path(__file__).parent))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='METRA dmc_cheetah_state discrete benchmark')
    p.add_argument('--exp_dir',      required=True,
                   help='Path to METRA experiment directory (contains option_policy*.pt)')
    p.add_argument('--epoch',        type=int, default=None,
                   help='Epoch number of checkpoint to load; defaults to latest')
    p.add_argument('--n_episodes',   type=int, default=1)
    p.add_argument('--episode_steps', type=int, default=1000)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--cuda_id',      type=int, default=0)
    p.add_argument('--out_dir',      type=str, default='./benchmark_results_metra_cheetah_discrete')
    return p.parse_args()


# ── Checkpoint loading ─────────────────────────────────────────────────────────

def find_latest_epoch(exp_dir):
    pts = list(Path(exp_dir).glob('option_policy*.pt'))
    if not pts:
        raise FileNotFoundError(f'No option_policy*.pt found in {exp_dir}')
    epochs = []
    for p in pts:
        stem = p.stem  # e.g. 'option_policy9000'
        try:
            epochs.append(int(stem.replace('option_policy', '')))
        except ValueError:
            pass
    if not epochs:
        raise FileNotFoundError(f'Cannot parse epoch numbers from {pts}')
    return max(epochs)


def load_policy(exp_dir, epoch, device):
    if epoch is None:
        epoch = find_latest_epoch(exp_dir)
    pt_path = Path(exp_dir) / f'option_policy{epoch}.pt'
    print(f'Loading policy: {pt_path}')
    if not pt_path.exists():
        raise FileNotFoundError(f'Checkpoint not found: {pt_path}')
    data = torch.load(pt_path, map_location=device)
    policy = data['policy'].to(device)
    policy.eval()
    dim_option = int(data['dim_option'])
    discrete = bool(data['discrete'])
    print(f'  dim_option={dim_option}, discrete={discrete}')
    assert discrete, 'This script is for discrete skills. Use benchmark_metra_cheetah_continuous.py for continuous.'
    return policy, dim_option, epoch


# ── Environment ────────────────────────────────────────────────────────────────

def make_env(seed):
    from envs.custom_dmc_tasks import dmc
    from garagei.envs.consistent_normalized_env import consistent_normalize
    env = dmc.make('cheetah_run_forward', obs_type='states', frame_stack=1, action_repeat=1, seed=seed)
    env = consistent_normalize(env, normalize_obs=False)
    return env


# ── Rollout ───────────────────────────────────────────────────────────────────

def rollout(policy, env, skill, episode_steps):
    """
    Run one episode with a fixed discrete skill (one-hot vector).
    X position from info['coordinates'][0] (torso x before each step).
    Returns:
        obs_seq : (T, obs_dim) float32
        x_seq   : (T,) float32
        ep_return: float
    """
    obs = env.reset()
    obs = np.asarray(obs, dtype=np.float32).flatten()
    concat_obs = np.concatenate([obs, skill]).astype(np.float32)

    obs_list, x_list = [], []
    ep_return = 0.0

    for _ in range(episode_steps):
        obs_list.append(obs.copy())

        action, _ = policy.get_mode_actions(concat_obs[None])
        action = action[0]

        next_obs, reward, done, info = env.step(action)
        x_list.append(float(info['coordinates'][0]))  # torso x before this step

        obs = np.asarray(next_obs, dtype=np.float32).flatten()
        concat_obs = np.concatenate([obs, skill]).astype(np.float32)
        ep_return += float(reward)
        if done:
            break

    return np.array(obs_list, dtype=np.float32), np.array(x_list, dtype=np.float32), ep_return


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_x_density(skill_x, n_skills, out_path, x_lim=None):
    cmap = plt.cm.get_cmap('hsv', n_skills)
    fig, ax = plt.subplots(figsize=(10, max(4, n_skills * 0.5 + 2)))

    for i in range(n_skills):
        x = skill_x[i]
        if len(x) < 2:
            continue
        try:
            kde = gaussian_kde(x)
            xs = np.linspace(x.min(), x.max(), 500)
            density = kde(xs)
            density = (density - density.min()) / (density.max() + 1e-8)
            y = i * 2
            ax.fill_between(xs, y + density, y - density, color=cmap(i), alpha=0.5)
        except Exception:
            pass

    ax.set_xlabel('X position (m)')
    ax.set_yticks([])
    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.set_title(f'METRA dmc_cheetah_state — X density per discrete skill\n{n_skills} skills')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')


def plot_cosine_matrix(embeddings, n_skills, out_path):
    dist_matrix = cosine_distances(embeddings)
    mean_dist = float(np.mean(dist_matrix[np.triu_indices(n_skills, k=1)]))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(dist_matrix, vmin=0, vmax=1, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Cosine distance')
    ax.set_title(f'Skill embedding cosine-distance matrix | mean = {mean_dist:.4f}')
    ax.set_xlabel('Skill index')
    ax.set_ylabel('Skill index')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {out_path}')
    return mean_dist, dist_matrix


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    policy, dim_option, epoch = load_policy(args.exp_dir, args.epoch, device)
    env = make_env(args.seed)

    # All discrete skills: np.eye(dim_option)
    all_skills = [np.eye(dim_option, dtype=np.float32)[i] for i in range(dim_option)]
    n_skills = dim_option
    print(f'\nBenchmarking {n_skills} discrete skills, {args.n_episodes} episode(s) each\n')

    skill_x     = {}   # i -> concatenated x array (for plotting)
    skill_x_eps = {}   # i -> list of per-episode x arrays (for CSV)
    skill_embed = {}   # i -> mean obs embedding
    skill_returns = {}

    for s_idx, skill in enumerate(all_skills):
        ep_obs_all, ep_x_all, ep_rets = [], [], []

        for _ in range(args.n_episodes):
            obs_seq, x_seq, ret = rollout(policy, env, skill, args.episode_steps)
            ep_obs_all.append(obs_seq)
            ep_x_all.append(x_seq)
            ep_rets.append(ret)

        skill_x[s_idx]      = np.concatenate(ep_x_all)
        skill_x_eps[s_idx]  = ep_x_all
        skill_embed[s_idx]  = np.mean(np.concatenate(ep_obs_all, axis=0), axis=0)
        skill_returns[s_idx] = ep_rets

        x_arr = skill_x[s_idx]
        print(f'  [{s_idx+1:2d}/{n_skills}] skill={s_idx}  '
              f'mean_return={np.mean(ep_rets):8.3f}  '
              f'x=[{x_arr.min():.1f}, {x_arr.max():.1f}]')

    all_x = np.concatenate(list(skill_x.values()))
    x_lim = (float(all_x.min()) - 1.0, float(all_x.max()) + 1.0)

    plot_x_density(skill_x, n_skills, out_dir / 'skill_x_density.png', x_lim=x_lim)

    embeddings = np.stack([skill_embed[i] for i in range(n_skills)])
    mean_dist, dist_matrix = plot_cosine_matrix(embeddings, n_skills, out_dir / 'cosine_distance_matrix.png')

    # ── X-location CSV ───────────────────────────────────────────────────────
    csv_path = out_dir / 'x_locations_metra_cheetah_discrete.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['skill_idx', 'episode', 'step', 'x'])
        for s_idx, eps in skill_x_eps.items():
            for ep_idx, x_seq in enumerate(eps):
                for step, x in enumerate(x_seq):
                    writer.writerow([s_idx, ep_idx, step, f'{x:.4f}'])
    print(f'Saved: {csv_path}')

    all_returns = [r for rs in skill_returns.values() for r in rs]
    x_range = all_x.max() - all_x.min()

    print('\n' + '='*55)
    print('METRA dmc_cheetah_state Discrete Benchmark Summary')
    print('='*55)
    print(f'  Checkpoint epoch    : {epoch}')
    print(f'  Skills              : {n_skills} discrete (one-hot)')
    print(f'  Episodes per skill  : {args.n_episodes}')
    print(f'  Episode steps       : {args.episode_steps}')
    print(f'  Mean return         : {np.mean(all_returns):.3f} ± {np.std(all_returns):.3f}')
    print(f'  Max / Min return    : {np.max(all_returns):.3f} / {np.min(all_returns):.3f}')
    print(f'  X coverage          : {x_range:.2f} m')
    print(f'  Cosine dist (mean)  : {mean_dist:.4f}')
    print(f'  Cosine dist (max)   : {dist_matrix[np.triu_indices(n_skills, k=1)].max():.4f}')
    print('='*55)


if __name__ == '__main__':
    main()
