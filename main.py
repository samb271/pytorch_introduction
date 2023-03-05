import torch
from torch import nn
import matplotlib.pyplot as plt
import LinearRegression
import sklearn
from sklearn.datasets import make_circles
import pandas as pd

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, 
                    noise = 0.03,
                    random_state=42)

print(f"First 5 samples of X: \n{X[:5]}")
print(f"First 5 samples of y: \n{y[:5]}")






