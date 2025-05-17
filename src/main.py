from src.environment.env_fatory import *
from src.config.configHandler import load_config


if __name__ == "__main__":
    cfg = load_config()
    envs = make_multiple_envs(cfg)
    print(cfg)
    print(envs)
