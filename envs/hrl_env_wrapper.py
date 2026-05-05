"""
Hierarchical RL environment wrapper for METRA.

The high-level policy selects a skill index; the low-level METRA PolicyEx
executes `low_level_steps` primitive actions with that skill fixed.

Supports:
  - HierarchicalMETRADiscreteEnv : skill_idx → one-hot → PolicyEx
"""

import gym
from gym import spaces
import numpy as np
import torch


class HierarchicalMETRADiscreteEnv(gym.Env):
    """
    High-level action: integer in [0, dim_option)  (skill index)
    Low-level actor : METRA PolicyEx, called with concat([obs, one_hot_skill])

    The policy is expected to already be on its target device.
    get_mode_actions() handles device movement internally.
    """

    def __init__(self, env, policy, dim_option, low_level_steps):
        super().__init__()
        self._env = env
        self.policy = policy
        self.dim_option = dim_option
        self.low_level_steps = low_level_steps

        # High-level chooses a skill index
        self.action_space = spaces.Discrete(dim_option)
        self.observation_space = env.observation_space

        self.last_obs = None

    # ------------------------------------------------------------------
    def reset(self):
        self.last_obs = np.asarray(self._env.reset(), dtype=np.float32)
        return self.last_obs.copy()

    def step(self, skill_idx):
        skill = np.eye(self.dim_option, dtype=np.float32)[int(skill_idx)]

        total_reward = 0.0
        obs = self.last_obs
        done = False
        info = {}

        for _ in range(self.low_level_steps):
            concat_obs = np.concatenate([obs, skill])          # (obs_dim + dim_option,)
            action, _ = self.policy.get_mode_actions(concat_obs[None])  # (1, act_dim)
            action = action[0].astype(np.float32)

            obs, reward, done, info = self._env.step(action)
            obs = np.asarray(obs, dtype=np.float32)
            total_reward += float(reward)
            if done:
                break

        self.last_obs = obs
        return obs.copy(), total_reward, done, info

    def seed(self, seed=None):
        if hasattr(self._env, 'seed'):
            self._env.seed(seed)

    def render(self, mode='human'):
        return self._env.render(mode=mode)

    def __getattr__(self, name):
        return getattr(self._env, name)
