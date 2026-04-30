# METRA: Scalable Unsupervised RL with Metric-Aware Abstraction

This repository contains the official implementation of **METRA: Scalable Unsupervised RL with Metric-Aware Abstraction**.
The implementation is based on
[Lipschitz-constrained Unsupervised Skill Discovery](https://github.com/seohongpark/LSD).

Visit [our project page](https://seohong.me/projects/metra/) for more results including videos.

## Requirements
- Python 3.8

## Installation

```
conda create --name metra python=3.8
conda activate metra
pip install -r requirements.txt --no-deps
pip install -e .
pip install -e garaged
```

## Examples

```
# METRA on state-based Ant (2-D skills)
python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 0 --dim_option 2

# LSD on state-based Ant (2-D skills)
python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --dual_reg 0 --spectral_normalization 1 --discrete 0 --dim_option 2

# DADS on state-based Ant (2-D skills)
python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo dads --inner 0 --unit_length 0 --dual_reg 0 --discrete 0 --dim_option 2

# DIAYN on state-based Ant (2-D skills)
python tests/main.py --run_group Debug --env ant --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --inner 0 --unit_length 0 --dual_reg 0 --discrete 0 --dim_option 2

# METRA on state-based HalfCheetah (16 skills)
python tests/main.py --run_group Debug --env half_cheetah --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type preset --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 1 --dim_option 16

# METRA on pixel-based Quadruped (4-D skills)
python tests/main.py --run_group Debug --env dmc_quadruped --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --encoder 1 --sample_cpu 0

# METRA on pixel-based Humanoid (2-D skills)
python tests/main.py --run_group Debug --env dmc_humanoid --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --frame_stack 3 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 1 --sample_cpu 0

# METRA on state-based Quadruped dm_control default (4-D skills)
python tests/main.py --run_group Debug --env dmc_quadruped_state --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 4 --encoder 0

# METRA on state-based Humanoid dm_control default (2-D skills)
python tests/main.py --run_group Debug --env dmc_humanoid_state --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 0

# METRA on pixel-based Kitchen (24 skills)
python tests/main.py --run_group Debug --env kitchen --max_path_length 50 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --num_video_repeats 1 --frame_stack 3 --sac_max_buffer_size 100000 --algo metra --sac_lr_a -1 --trans_optimization_epochs 100 --n_epochs_per_log 25 --n_epochs_per_eval 250 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 1 --dim_option 24 --encoder 1 --sample_cpu 0

# METRA on Ant-v5 gymnasium (2-D skills)
python tests/main.py --run_group Debug --env ant_v5 --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type off --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo metra --discrete 0 --dim_option 2

# RSD on Ant-v5 gymnasium (2-D skills)
python tests/main.py --run_group Debug --env ant_v5 --max_path_length 200 --seed 0 --traj_batch_size 8 --n_parallel 1 --normalizer_type off --eval_plot_axis -50 50 -50 50 --trans_optimization_epochs 50 --n_epochs_per_log 100 --n_epochs_per_eval 1000 --n_epochs_per_save 10000 --sac_max_buffer_size 1000000 --algo rsd --explore_type SZN --phi_type baseline --policy_type baseline --discrete 0 --dim_option 2

# METRA on AntMaze-UMaze (2-D skills) — obs_type=states, action_repeat=1, matches TIME/DUSDi default
python tests/main.py --run_group Debug --env antmaze_umaze --max_path_length 700 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 1000000 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 0

# RSD on AntMaze-UMaze (2-D skills) — obs_type=states, action_repeat=1, matches TIME/DUSDi default
python tests/main.py --run_group Debug --env antmaze_umaze --max_path_length 700 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 1000000 --algo rsd --explore_type SZN --phi_type baseline --policy_type baseline --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 0

# METRA on AntMaze-Medium-Play (2-D skills)
python tests/main.py --run_group Debug --env antmaze_medium_play --max_path_length 1000 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 1000000 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 0

# METRA on AntMaze-Large-Play (2-D skills)
python tests/main.py --run_group Debug --env antmaze_large_play --max_path_length 1000 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --sac_max_buffer_size 1000000 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 0

# METRA on state-based Cheetah-run-forward dm_control (2-D skills) — env config matches TIME pretrain default
python tests/main.py --run_group Debug --env dmc_cheetah_state --max_path_length 1000 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 0

# RSD on state-based Cheetah-run-forward dm_control (2-D skills) — env config matches TIME pretrain default
python tests/main.py --run_group Debug --env dmc_cheetah_state --max_path_length 1000 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo rsd --explore_type SZN --phi_type baseline --policy_type baseline --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 0

# METRA on state-based Hopper-hop dm_control (2-D skills) — env config matches TIME pretrain default
python tests/main.py --run_group Debug --env dmc_hopper_state --max_path_length 1000 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo metra --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 0

# RSD on state-based Hopper-hop dm_control (2-D skills) — env config matches TIME pretrain default
python tests/main.py --run_group Debug --env dmc_hopper_state --max_path_length 1000 --seed 0 --traj_batch_size 8 --n_parallel 4 --normalizer_type off --video_skip_frames 2 --sac_max_buffer_size 300000 --eval_plot_axis -15 15 -15 15 --algo rsd --explore_type SZN --phi_type baseline --policy_type baseline --trans_optimization_epochs 200 --n_epochs_per_log 25 --n_epochs_per_eval 125 --n_epochs_per_save 1000 --n_epochs_per_pt_save 1000 --discrete 0 --dim_option 2 --encoder 0
```

## Hopper-hop Environment Details

The `dmc_hopper_state` environment wraps the standard `dm_control` suite `hopper_hop` task with the following config, identical to the default pre-training config in the TIME repo:

| Parameter | Value |
|---|---|
| Task | `hopper_hop` (dm_control suite) |
| `obs_type` | `states` |
| `action_repeat` | 1 |
| `frame_stack` | 1 (no stacking) |
| `flat_observation` | True |
| Episode length | 1000 steps (20 s / 0.02 s per step) |
| `dim_option` | 2 |
| `normalizer_type` | off |

## AntMaze Environment Details

The `antmaze_*` environments wrap `gymnasium-robotics` AntMaze_*-v5 with observation layout matching TIME and DUSDi:

| Parameter | Value |
|---|---|
| Library | `gymnasium-robotics` (`AntMaze_*-v5`) |
| `obs_type` | `states` only |
| Observation | `observation(27) + achieved_goal(2) + desired_goal(2)` = 31 dims |
| `action_repeat` | 1 |
| `dim_option` | 2 |
| `normalizer_type` | off |
| `max_path_length` | 700 (umaze) / 1000 (medium, large) |

**Install prerequisite:**
```bash
pip install gymnasium-robotics
```

**Supported variants:** `antmaze_umaze`, `antmaze_medium_play`, `antmaze_medium_diverse`, `antmaze_large_play`, `antmaze_large_diverse`
