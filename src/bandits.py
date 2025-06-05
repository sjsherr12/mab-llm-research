import numpy as np

# Bernoulli Distribution Bandit Class
class BernoulliBandit:
    def __init__(self, probs):
        """
        probs: list of success probabilities for each arm (e.g. [0.3, 0.5, 0.7])
        """
        self.K = len(probs)
        self.probs = probs

    def pull(self, arm):
        """
        Pull an arm and return 0 or 1 reward.
        """
        return np.random.rand() < self.probs[arm]

# Gaussian Distribution Bandit Class
class GaussianBandit:
    def __init__(self, mus, sigmas):
        """
        mus: list of mean rewards for each arm
        sigmas: list of standard deviations for each arm
        """
        self.K = len(mus)
        self.mus = mus
        self.sigmas = sigmas

    def pull(self, arm):
        """
        Pull an arm and return a sample from N(mu, sigma^2).
        """
        return np.random.normal(self.mus[arm], self.sigmas[arm])
