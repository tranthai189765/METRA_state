#!/usr/bin/env python3
"""
train_hrl.py — HRL downstream training for METRA discrete skills

Loads a pretrained METRA option_policy checkpoint (discrete skills) and trains
a high-level PPO policy on a downstream task using SB3.

Architecture:
  High-level : PPO  (selects skill index every --low_level_steps env steps)
  Low-level  : frozen METRA PolicyEx  (executes primitive actions)

Usage:
    python train_hrl.py \\
        --exp_dir /workspace/METRA_state/exp/Debug/sd001_1777660191_dmc_cheetah_state_metra \\
        --env dmc_cheetah_state \\
        --ds_task cheetah_run \\
        --dim_option 16 \\
        --low_level_steps 50 \\
        --n_env 8 \\
        --n_steps 256 \\
        --total_timesteps 50000000 \\
        --seed 1 \\
        --use_tb \\
        --out_dir ./hrl_results/metra_cheetah_run
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('MUJOCO_GL', 'egl')

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor

sys.path.insert(0, str(Path(__file__).parent))


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--exp_dir',         required=True,
                   help='METRA experiment dir containing option_policy*.pt')
    p.add_argument('--epoch',           type=int,   default=None,
                   help='Checkpoint epoch; auto-detects latest if omitted')
    p.add_argument('--env',             type=str,   default='dmc_cheetah_state',
                   choices=['dmc_cheetah_state', 'dmc_quadruped_state',
                            'dmc_humanoid_state', 'dmc_hopper_state'],
                   help='Pretraining environment')
    p.add_argument('--ds_task',         type=str,   default='cheetah_run',
                   help='Downstream dm_control task, e.g. cheetah_run, quadruped_walk')
    p.add_argument('--dim_option',      type=int,   default=16,
                   help='Number of discrete skills (must match checkpoint)')
    p.add_argument('--low_level_steps', type=int,   default=50,
                   help='Low-level env steps per high-level skill selection')
    p.add_argument('--episode_steps',   type=int,   default=1000,
                   help='Max env steps per episode')
    p.add_argument('--n_env',           type=int,   default=8,
                   help='Number of parallel envs (1 = no subprocess)')
    p.add_argument('--n_steps',         type=int,   default=256,
                   help='PPO rollout steps per env per update')
    p.add_argument('--total_timesteps', type=int,   default=50_000_000)
    p.add_argument('--seed',            type=int,   default=1)
    p.add_argument('--cuda_id',         type=int,   default=0)
    p.add_argument('--use_tb',          action='store_true',
                   help='Enable TensorBoard logging')
    p.add_argument('--out_dir',         type=str,   default='./hrl_results')
    return p.parse_args()


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def find_latest_epoch(exp_dir):
    pts = list(Path(exp_dir).glob('option_policy*.pt'))
    if not pts:
        raise FileNotFoundError(f'No option_policy*.pt in {exp_dir}')
    epochs = []
    for pt in pts:
        try:
            epochs.append(int(pt.stem.replace('option_policy', '')))
        except ValueError:
            pass
    if not epochs:
        raise FileNotFoundError(f'Cannot parse epoch from filenames in {exp_dir}')
    return max(epochs)


def load_policy_cpu(exp_dir, epoch):
    """Load PolicyEx onto CPU (safe for SubprocVecEnv pickling)."""
    if epoch is None:
        epoch = find_latest_epoch(exp_dir)
    pt_path = Path(exp_dir) / f'option_policy{epoch}.pt'
    print(f'Loading policy: {pt_path}')
    data = torch.load(pt_path, map_location='cpu')
    policy = data['policy'].cpu()
    policy.eval()
    assert bool(data['discrete']), \
        'This script expects a discrete-skill checkpoint (--discrete 1 during pretraining).'
    dim_option = int(data['dim_option'])
    return policy, dim_option, epoch


# ── Environment factory ────────────────────────────────────────────────────────

# Maps env name → (dmc domain, default pretraining task)
_ENV_TASK_MAP = {
    'dmc_cheetah_state':   ('cheetah',  'run'),
    'dmc_quadruped_state': ('quadruped', 'run'),
    'dmc_humanoid_state':  ('humanoid',  'run'),
    'dmc_hopper_state':    ('hopper',    'hop'),
}

_DS_TASK_MAP = {
    # cheetah
    'cheetah_run':           'cheetah_run_forward',
    'cheetah_run_backward':  'cheetah_run_backward',
    # quadruped
    'quadruped_walk':        'quadruped_walk',
    'quadruped_run':         'quadruped_run',
    # humanoid
    'humanoid_run':          'humanoid_run',
    # hopper
    'hopper_hop':            'hopper_hop',
    'hopper_stand':          'hopper_stand',
}


def _make_low_level_env(env_name, ds_task, episode_steps, seed):
    """Create the primitive-action env for the low-level controller."""
    from envs.custom_dmc_tasks import dmc
    from garagei.envs.consistent_normalized_env import consistent_normalize

    dmc_task = _DS_TASK_MAP.get(ds_task)
    if dmc_task is None:
        # Fallback: try using ds_task directly as a dmc task string
        dmc_task = ds_task

    env = dmc.make(dmc_task, obs_type='states', frame_stack=1, action_repeat=1, seed=seed)
    env = consistent_normalize(env, normalize_obs=False)
    return env


def make_hrl_env_factory(exp_dir, epoch, env_name, ds_task, dim_option,
                         low_level_steps, episode_steps, seed):
    """
    Returns a no-arg callable that creates one HRL env.
    The policy is reloaded from disk in each call (subprocess-safe).
    """
    from envs.hrl_env_wrapper import HierarchicalMETRADiscreteEnv

    def _make(rank=0):
        policy, _, _ = load_policy_cpu(exp_dir, epoch)
        env = _make_low_level_env(env_name, ds_task, episode_steps, seed + rank)
        hrl_env = HierarchicalMETRADiscreteEnv(env, policy, dim_option, low_level_steps)
        return hrl_env

    return _make


# ── SB3 callback ───────────────────────────────────────────────────────────────

class EpisodeCountCallback(BaseCallback):
    """Track cumulative episodes so CSV x-axis is episodes, not steps."""

    def __init__(self):
        super().__init__()
        self._ep_count = 0

    def _on_step(self) -> bool:
        for info in self.locals.get('infos', []):
            if 'episode' in info:
                self._ep_count += 1
        return True

    def _on_rollout_end(self) -> None:
        self.logger.record('time/episodes', self._ep_count)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_random_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda_id}' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect epoch once so all workers load the same checkpoint
    epoch = args.epoch if args.epoch is not None else find_latest_epoch(args.exp_dir)
    print(f'Using checkpoint epoch {epoch}')

    # Verify dim_option matches checkpoint
    _, ckpt_dim, _ = load_policy_cpu(args.exp_dir, epoch)
    if ckpt_dim != args.dim_option:
        print(f'Warning: checkpoint dim_option={ckpt_dim} != --dim_option={args.dim_option}. '
              f'Using checkpoint value {ckpt_dim}.')
        args.dim_option = ckpt_dim

    make_env = make_hrl_env_factory(
        exp_dir=args.exp_dir,
        epoch=epoch,
        env_name=args.env,
        ds_task=args.ds_task,
        dim_option=args.dim_option,
        low_level_steps=args.low_level_steps,
        episode_steps=args.episode_steps,
        seed=args.seed,
    )

    # Build vectorised env
    if args.n_env == 1:
        train_env = VecMonitor(DummyVecEnv([lambda: make_env(rank=0)]))
    else:
        env_fns = [lambda rank=i: make_env(rank=rank) for i in range(args.n_env)]
        train_env = VecMonitor(SubprocVecEnv(env_fns))

    # SB3 logger
    log_formats = ['stdout', 'csv']
    if args.use_tb:
        log_formats.append('tensorboard')
    sb3_log = sb3_configure(str(out_dir), log_formats)

    model = PPO('MlpPolicy', train_env, verbose=1,
                n_steps=args.n_steps, device=device)
    model.set_logger(sb3_log)

    print(f'\nHRL training: METRA discrete ({args.dim_option} skills) → {args.ds_task}')
    print(f'  low_level_steps={args.low_level_steps}, n_env={args.n_env}, '
          f'n_steps={args.n_steps}, total={args.total_timesteps:,}\n')

    model.learn(total_timesteps=args.total_timesteps, callback=EpisodeCountCallback())
    model.save(str(out_dir / 'ppo_weight'))
    print(f'\nSaved PPO weights to {out_dir / "ppo_weight.zip"}')


if __name__ == '__main__':
    main()
