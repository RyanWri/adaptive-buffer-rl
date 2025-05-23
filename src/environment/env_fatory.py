import gymnasium as gym


def make_env(cfg: dict, env_name: str):
    kwargs = {k: v for k, v in cfg.items() if k != "env_name"}
    return gym.make(env_name, **kwargs)

def make_multiple_envs(cfg: dict) -> dict:
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
