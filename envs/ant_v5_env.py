import numpy as np
from gym import spaces, core

from envs.mujoco.mujoco_utils import MujocoTrait


class AntV5Env(MujocoTrait, core.Env):
    """Gymnasium Ant-v5 wrapped to old-gym API for METRA.

    Observation: 27-dim (qpos[2:] + qvel, no contact forces by default).
    Coordinates: (x_position, y_position) from Ant-v5 info dict.
    Action space: 8-dim, already in [-1, 1].
    """

    def __init__(self, seed=0, render_hw=100):
        try:
            import gymnasium
        except ImportError as e:
            raise ImportError(
                'gymnasium is required for Ant-v5. '
                'Install with: pip install gymnasium[mujoco]'
            ) from e

        self._env = gymnasium.make('Ant-v5', render_mode=None)
        self._seed = seed
        self._seeded = False
        self.render_hw = render_hw

        # Track torso position for coordinates (not in obs, comes from info)
        self._x_pos = 0.0
        self._y_pos = 0.0

        obs, _ = self._env.reset(seed=seed)
        self._seeded = True
        obs = obs.astype(np.float32)

        act = self._env.action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=act.low.astype(np.float32),
            high=act.high.astype(np.float32),
            dtype=np.float32,
        )

    def reset(self):
        if not self._seeded:
            obs, info = self._env.reset(seed=self._seed)
            self._seeded = True
        else:
            obs, info = self._env.reset()
        self._x_pos = float(info.get('x_position', 0.0))
        self._y_pos = float(info.get('y_position', 0.0))
        return obs.astype(np.float32)

    def step(self, action, render=False):
        x_before = self._x_pos
        y_before = self._y_pos

        obs, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated

        self._x_pos = float(info.get('x_position', x_before))
        self._y_pos = float(info.get('y_position', y_before))

        out_info = {
            'coordinates': np.array([x_before, y_before], dtype=np.float32),
            'next_coordinates': np.array([self._x_pos, self._y_pos], dtype=np.float32),
            'ori_obs': obs.astype(np.float32),
            'next_ori_obs': obs.astype(np.float32),
            'terminated': terminated,
        }

        if render:
            frame = self._env.render()
            if frame is not None:
                out_info['render'] = np.asarray(frame).transpose(2, 0, 1)

        return obs.astype(np.float32), float(reward), done, out_info

    def seed(self, seed=None):
        pass

    def render(self, mode='rgb_array', **kwargs):
        return self._env.render()

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        return super().calc_eval_metrics(
            trajectories, is_option_trajectories, coord_dims=[0, 1]
        )
