import numpy as np
from src.bandits import BernoulliBandit, GaussianBandit

# Bernoulli Bandit Demo
np.random.seed(42)
bernoulli_bandit = BernoulliBandit([0.3, 0.5, 0.7])
print("Bernoulli Bandit Demo:")
for i in range(10):
    arm = np.random.randint(0, 3)
    reward = bernoulli_bandit.pull(arm)
    print(f"Pulled arm {arm}, got reward {reward}")

print("\nGaussian Bandit Demo:")
# Gaussian Bandit Demo
np.random.seed(42)
gaussian_bandit = GaussianBandit([1.0, 1.5, 2.0], [0.1, 0.1, 0.1])
for i in range(10):
    arm = np.random.randint(0, 3)
    reward = gaussian_bandit.pull(arm)
    print(f"Pulled arm {arm}, got reward {reward:.2f}")
