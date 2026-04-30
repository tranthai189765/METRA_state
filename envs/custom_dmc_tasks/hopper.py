from dm_control import suite


def make(task, task_kwargs=None, environment_kwargs=None, visualize_reward=False):
    task_kwargs = task_kwargs or {}
    env = suite.load(
        'hopper',
        task,
        task_kwargs=task_kwargs,
        environment_kwargs=environment_kwargs or {},
        visualize_reward=visualize_reward,
    )
    return env
