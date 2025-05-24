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
