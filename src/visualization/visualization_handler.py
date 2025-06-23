import numpy as np
import matplotlib.pyplot as plt


def moving_average(data, window=100):
    """
    Computes the moving average of a list/array.

    Params:
        :data: list or numpy array of episode rewards
        :window: size of the moving average window

    Returns:
        smoothed moving average array
    """
    return np.convolve(data, np.ones(window)/window, mode='valid')


def plot_reward_over_episodes(reward_list, algorithm, window):
    """
    Plots the smoothed episode reward over time for a given reinforcement learning algorithm.

    Params:
        :reward_list: A list of episodic rewards collected during training.
        :algorithm: The name of the algorithm being evaluated (used in the plot title).
        :window: The window size for the moving average filter.
    """
    smoothed = moving_average(reward_list, window)
    plt.figure(figsize=(10, 5))
    plt.plot(smoothed)
    plt.xlabel("Episode")
    plt.ylabel(f"Average Reward (window={window})")
    plt.title(f"Smoothed Episode Reward Over Time - {algorithm}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_side_by_side(game_id, aper_rewards, vanilla_rewards, window):
    plt.figure(figsize=(12, 6))
    plt.title(f'Game {game_id}')

    # Plot raw episode rewards (optional, can be commented for less clutter)
    plt.plot(aper_rewards, alpha=0.2, label="BETDQNet (raw)")
    plt.plot(aper_rewards, alpha=0.2, label="VanillaDQN (raw)")

    # Plot moving averages for both curves
    plt.plot(moving_average(aper_rewards, window), label="BETDQNet (smoothed)")
    plt.plot(moving_average(vanilla_rewards, window), label="VanillaDQN (smoothed)")

    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Episode Reward Comparison: BETDQNet vs VanillaDQN")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
