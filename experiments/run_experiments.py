from src.bandits import BernoulliBandit
from src.algorithms import run_epsilon_greedy
import matplotlib.pyplot as plt

def main():
    bandit = BernoulliBandit([0.1, 0.5, 0.9])
    T = 500
    regret_curve = []

    cumulative_rewards = run_epsilon_greedy(bandit, T, epsilon=0.1)
    optimal_reward = T * bandit.probs[bandit.optimal_arm()]
    regret = optimal_reward - cumulative_rewards

    plt.plot(regret, label='ε-greedy')
    plt.xlabel("Time")
    plt.ylabel("Cumulative Regret")
    plt.legend()
    plt.title("Regret Curve")
    plt.savefig("figures/epsilon_greedy_regret.png")
    plt.show()

if __name__ == "__main__":
    main()