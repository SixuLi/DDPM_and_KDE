from numpy.linalg import norm as L2
from scipy.stats import norm as univariate_normal
from scipy.stats import multivariate_normal
from tqdm import tqdm_notebook
import numpy as np

class KernelDensityEstimator:
    def __init__(self, train_data):
        self.train_data = train_data

    def est_bandwidth(self):
        # Estimate bandwidth using scott's rule

        # Retrieve data
        data = self.train_data

        # Get number of samples
        n = data.shape[0]

        # Get dim of data
        d = data.shape[1]

        # Compute standard deviation
        std = np.std(data, axis=0)

        # Construct the H diagonal bandwidth matrix with std along the diag
        H = (n ** (-1/(d+4)) * np.diag(std)) ** 2

        return H

    def random_sample(self,scaling_factor):
        # Randomly generate a new sample from the dataset

        # Get H
        H = self.est_bandwidth() * scaling_factor

        # Randomly pick a data point
        random_data = np.random.permutation(self.train_data)[0]

        # Sample
        sample = np.random.multivariate_normal(mean=random_data,cov=H)

        return random_data, sample

    def explicit_sample(self, scaling_factor, data):
        # Get H
        H = self.est_bandwidth() * scaling_factor

        # Sample
        sample = np.random.multivariate_normal(mean=data, cov=H)

        return data, sample