import numpy as np

# Epsilon-Greedy Algorithm
def run_epsilon_greedy(bandit, epsilon, steps):
    K = bandit.K
    counts = np.zeros(K)
    values = np.zeros(K)
    regrets = []
    optimal_mean = max(bandit.probs)

    for t in range(steps):
        if np.random.rand() < epsilon:
            arm = np.random.randint(K)
        else:
            arm = np.argmax(values)

        reward = bandit.pull(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]  # update mean

        regret = optimal_mean - bandit.probs[arm]
        regrets.append(regret)

    return np.cumsum(regrets)

# UCB1 Algorithm
def run_ucb(bandit, steps):
    K = bandit.K
    counts = np.zeros(K)
    values = np.zeros(K)
    regrets = []
    optimal_mean = max(bandit.probs)

    for t in range(steps):
        if t < K:
            arm = t  # pull each arm once
        else:
            ucb_values = values + np.sqrt(2 * np.log(t + 1) / (counts + 1e-5))
            arm = np.argmax(ucb_values)

        reward = bandit.pull(arm)
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]

        regret = optimal_mean - bandit.probs[arm]
        regrets.append(regret)

    return np.cumsum(regrets)

# Thompson Sampling Algorithm (Bernoulli)
def run_thompson_sampling(bandit, steps):
    K = bandit.K
    alpha = np.ones(K)
    beta = np.ones(K)
    regrets = []
    optimal_mean = max(bandit.probs)

    for t in range(steps):
        samples = np.random.beta(alpha, beta)
        arm = np.argmax(samples)
        reward = bandit.pull(arm)

        alpha[arm] += reward
        beta[arm] += 1 - reward

        regret = optimal_mean - bandit.probs[arm]
        regrets.append(regret)

    return np.cumsum(regrets)
