import numpy as np
from gym import spaces, core

from envs.mujoco.mujoco_utils import MujocoTrait

_TASK_MAP = {
    'antmaze_umaze':           'AntMaze_UMaze-v5',
    'antmaze_medium_play':     'AntMaze_Medium_Play-v5',
    'antmaze_medium_diverse':  'AntMaze_Medium_Diverse-v5',
    'antmaze_large_play':      'AntMaze_Large_Play-v5',
    'antmaze_large_diverse':   'AntMaze_Large_Diverse-v5',
}


class AntMazeEnv(MujocoTrait, core.Env):
    """Gymnasium-robotics AntMaze wrapped to old-gym API for METRA.

    Observation order matches TIME / DUSDi:
        observation (27) + achieved_goal (2) + desired_goal (2) = 31 dims.
    Coordinates tracked = achieved_goal (x, y position of the ant in maze).
    """

    def __init__(self, task, seed=0):
        try:
            import gymnasium
            import gymnasium_robotics  # noqa: F401 — registers AntMaze_*-v5 envs
        except ImportError as e:
            raise ImportError(
                'gymnasium-robotics is required for AntMaze. '
                'Install with: pip install gymnasium-robotics'
            ) from e

        env_id = _TASK_MAP[task]
        # render_mode=None avoids EGL context issues when forked in multiprocessing.
        self._env = gymnasium.make(env_id, render_mode=None)
        self._seed = seed
        self._seeded = False
        self._last_obs_dict = None

        obs_dict, _ = self._env.reset(seed=seed)
        self._seeded = True
        self._last_obs_dict = obs_dict
        flat_obs = self._flatten_obs(obs_dict)

        act = self._env.action_space
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=flat_obs.shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=act.low.astype(np.float32),
            high=act.high.astype(np.float32),
            dtype=np.float32,
        )

    def _flatten_obs(self, obs_dict):
        # Order matches TIME's GymnasiumWrapper and DUSDi's AntMazeGymEnv
        parts = []
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key in obs_dict:
                parts.append(np.asarray(obs_dict[key], dtype=np.float32).ravel())
        return np.concatenate(parts)

    def reset(self):
        if not self._seeded:
            obs_dict, _ = self._env.reset(seed=self._seed)
            self._seeded = True
        else:
            obs_dict, _ = self._env.reset()
        self._last_obs_dict = obs_dict
        return self._flatten_obs(obs_dict)

    def step(self, action, render=False):
        xy_before = np.array(self._last_obs_dict['achieved_goal'], dtype=np.float32)
        obs_dict, reward, terminated, truncated, info = self._env.step(action)
        done = terminated or truncated
        xy_after = np.array(obs_dict['achieved_goal'], dtype=np.float32)
        self._last_obs_dict = obs_dict

        if not isinstance(info, dict):
            info = {}
        info['coordinates'] = xy_before          # (2,) x,y before step
        info['next_coordinates'] = xy_after       # (2,) x,y after step
        info['ori_obs'] = self._flatten_obs(self._last_obs_dict)
        info['terminated'] = terminated

        if render:
            frame = self._env.render()
            if frame is not None:
                info['render'] = np.asarray(frame).transpose(2, 0, 1)

        return self._flatten_obs(obs_dict), float(reward), done, info

    def seed(self, seed=None):
        pass

    def render(self, mode='rgb_array', **kwargs):
        return self._env.render()

    def calc_eval_metrics(self, trajectories, is_option_trajectories):
        return super().calc_eval_metrics(
            trajectories, is_option_trajectories, coord_dims=[0, 1]
        )
