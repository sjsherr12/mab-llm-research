import numpy as np
import matplotlib.pyplot as plt

from src.bandits import BernoulliBandit, GaussianBandit
from src.algorithms import run_epsilon_greedy, run_ucb, run_thompson_sampling

def run_multiple_trials(algorithm_fn, bandit_class, bandit_args, steps, trials, **kwargs):
    regrets = np.zeros(steps)
    for seed in range(trials):
        # Ensure seed overrides or is injected into the bandit_args dictionary
        bandit_args_with_seed = dict(bandit_args)
        bandit_args_with_seed["seed"] = seed

        bandit = bandit_class(**bandit_args_with_seed)
        regrets += algorithm_fn(bandit, steps, **kwargs)
    return regrets / trials

def run_and_plot(bandit_class, bandit_args, steps, trials, epsilon, bandit_name, filename, skip_thompson_sampling=False):
    import matplotlib.pyplot as plt
    from src.algorithms import run_epsilon_greedy, run_ucb, run_thompson_sampling
    
    regret_egreedy = run_multiple_trials(run_epsilon_greedy, bandit_class, bandit_args, steps, trials, epsilon=epsilon)
    regret_ucb = run_multiple_trials(run_ucb, bandit_class, bandit_args, steps, trials)
    
    plt.plot(regret_egreedy, label='ε-greedy')
    plt.plot(regret_ucb, label='UCB')

    if not skip_thompson_sampling:
        regret_ts = run_multiple_trials(run_thompson_sampling, bandit_class, bandit_args, steps, trials)
        plt.plot(regret_ts, label='Thompson Sampling')

    plt.xlabel("Time")
    plt.ylabel("Cumulative Regret")
    plt.title(f"Regret Curves for {bandit_name}")
    plt.legend()
    plt.savefig(f"figures/{filename}")
    plt.show()

def main():
    # Bernoulli Bandit setup and runs
    run_and_plot(
        bandit_class=BernoulliBandit,
        bandit_args={"probs": [0.3, 0.5, 0.7], "seed": 42},
        steps=500,
        trials=100,
        epsilon=0.1,
        bandit_name="Bernoulli, K=3",
        filename="bernoulli_regret.png"
    )

    # Gaussian Bandit setup and runs (skip Thompson Sampling here)
    run_and_plot(
        bandit_class=GaussianBandit,
        bandit_args={"mus": [1.0, 1.5, 2.0], "sigmas": [0.1, 0.1, 0.1], "seed": 42},
        steps=500,
        trials=100,
        epsilon=0.1,
        bandit_name="Gaussian, K=3",
        filename="gaussian_regret.png",
        skip_thompson_sampling=True  # new flag to skip TS
    )

if __name__ == "__main__":
    main()