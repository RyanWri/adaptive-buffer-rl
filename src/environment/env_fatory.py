import gymnasium as gym


def make_env(cfg: dict, env_name: str):
    """
    Creates a Gymnasium environment.

    Params:
        :cfg: The environment config
        :env_name: The name of the desired gymnasium environment.

    Returns:
        A gymnasium environment with the desired configurations.
    """
    kwargs = {k: v for k, v in cfg.items() if k != "env_name"}
    return gym.make(env_name, **kwargs)

def make_multiple_envs(cfg: dict) -> dict:
    """
    Creates a list of Gymnasium environments.

    Params:
        :cfg: The config from config.yaml file.

    raises ValueError if the environment config is not a dictionary or if the config.yaml file is missing 'environments'.
    raises KeyError if the 'env_name' field is missing from the environment's config.

    Returns:
        A list of gymnasium environments with the desired configurations.
    """
    envs_cfg = cfg.get("environments")
    if not isinstance(envs_cfg, dict):
        raise ValueError(
            "Config must have an 'environments' dict; "
            f"got {type(envs_cfg).__name__}"
        )

    envs = {}
    for key, env_cfg in envs_cfg.items():
        if not isinstance(env_cfg, dict):
            raise ValueError(f"Environment '{key}' must be a dict; got {type(env_cfg).__name__}")
        if "env_name" not in env_cfg:
            raise KeyError(f"Environment '{key}' is missing required 'env_name' field")

        name = env_cfg["env_name"]
        params = {k: v for k, v in env_cfg.items() if k != "env_name"}
        envs[key] = gym.make(name, **params)
    return envs
