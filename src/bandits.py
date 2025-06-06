import numpy as np

# Bernoulli Distribution Bandit Class
class BernoulliBandit:
    def __init__(self, probs, seed=None):
        """
        probs: list of success probabilities for each arm (e.g. [0.3, 0.5, 0.7])
        """
        if seed is not None:
            np.random.seed(seed)

        self.K = len(probs)
        self.probs = probs

    def pull(self, arm):
        """
        Pull an arm and return 0 or 1 reward.
        """
        return int(np.random.rand() < self.probs[arm])
    
    def optimal_reward(self):
        """
        Return best reward that could've been obtained
        """
        return max(self.probs)
    
    def mean_reward(self, arm):
        return self.probs[arm]  # in Bernoulli

# Gaussian Distribution Bandit Class
class GaussianBandit:
    def __init__(self, mus, sigmas, seed=None):
        """
        mus: list of mean rewards for each arm
        sigmas: list of standard deviations for each arm
        """

        if seed is not None:
            np.random.seed(seed)

        self.K = len(mus)
        self.mus = mus
        self.sigmas = sigmas

    def pull(self, arm):
        """
        Pull an arm and return a sample from N(mu, sigma^2).
        """
        return np.random.normal(self.mus[arm], self.sigmas[arm])
    
    def optimal_reward(self):
        """
        Return the best expected reward (max mean).
        """
        return max(self.mus)
    
    def mean_reward(self, arm):
        return self.mus[arm]  # in Gaussian