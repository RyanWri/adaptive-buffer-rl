from src.environment.env_fatory import *
from src.config.configHandler import load_config
from src.BETDQNet.BETDQNet import run_BETDQNet
from src.BETDQNet.VanillaDQN import run_VanillaDQN
from src.visualization.visualization_handler import plot_reward_over_episodes

hyper_params = {"episodes": 250,
                "batch_size": 64,
                "gamma": 0.99,
                "discount_factor": 0.99,
                "learning_rate": 0.001,
                "memory_size": 10_000,
                "epsilon_start": 1.0,
                "epsilon_end": 0.1,
                "epsilon_decay_period": 5_000,
                "train_start": 1_000,
                "w1": 0.2,
                "w2": 0.8,
                "zeta": 2.2}


def init_conf_params(config: dict):
    params = config.get("parameters")
    if not params:
        print("No parameters set in the config.yaml file! Leaving as default settings.")
        return
    for param_key, value in params.items():
        if param_key in hyper_params.keys():
            hyper_params[param_key] = value
        else:
            print(f"Param name: {param_key} does not exist in hyper parameters config. No action taken.")


if __name__ == "__main__":
    # Load configurations from config.yaml file.
    cfg = load_config()
    # Initialize parameters (if any were added to the config.yaml file under 'parameters' key.
    init_conf_params(cfg)

    envs_cfg = cfg.get("environments", {})
    if not envs_cfg:
        raise ValueError("No 'environments' section found in your config.")

    # Start run on each environment.
    for key, env_cfg in envs_cfg.items():
        if not isinstance(env_cfg, dict):
            raise ValueError(f"Environment '{key}' must be a dict; got {type(env_cfg).__name__}")
        if "env_name" not in env_cfg:
            raise KeyError(f"Environment '{key}' is missing required 'env_name' field")

        env_name = env_cfg["env_name"]
        print(f"Running Environment: {env_name}")

        env = make_env(env_cfg, env_name)
        # Run the BETDQNet on the current environment.
        betdqnet_rewards = run_BETDQNet(env, hyper_params, env_cfg)

        # The environment is being closed at the end for BETDQNet so we create a new environment for VanillaDQN.
        env = make_env(env_cfg, env_name)
        # Run the VanillaDQN on the current environment.
        vanillaDQN_rewards = run_VanillaDQN(env, hyper_params, env_cfg)

        plot_reward_over_episodes(betdqnet_rewards, "BETDQNet", window=5)
        plot_reward_over_episodes(vanillaDQN_rewards, "VanillaDQN", window=5)




