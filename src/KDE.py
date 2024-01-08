import numpy as np
import os
from torchvision import datasets
from PIL import Image


class KernelDensityEstimator:
    def __init__(self, dataset_name, args, train_init):
        self.dataset_name = dataset_name
        self.args = args
        self.train_init = train_init
        self.initilization()

    def initilization(self):
        if self.dataset_name == 'CIFAR10':
            # Download the dataset
            dataset = datasets.CIFAR10(root='./results', train=True, download=True)
            data = dataset.data

            # Flatten
            data = data.reshape(data.shape[0], -1)

            self.train_data = data

            print('Finish image transform.')

    def inverse_transform(self, x):
        # return self.scaler_obj.inverse_transform(np.expand_dims(x,0))[0].reshape(32,32,3)
        # return self.pca_obj.inverse_transform(self.scaler_obj.inverse_transform(np.expand_dims(x, 0)))[0].reshape(32, 32, 3)
        return x.reshape(32,32,3)

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
        print('Begin sample generations!')

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

    def visualization(self, sample, tag):
        sample = self.inverse_transform(sample)
        sample = sample.clip(0,255).astype('uint8')
        if tag == 'original':
            figsave_path = os.path.join(self.train_init.output_path, 'kde_sampling_original_cifar10_' + '{}.png'.format(self.args.seed))
        elif tag == 'KDE_generate':
            figsave_path = os.path.join(self.train_init.output_path, 'kde_sampling_sample_cifar10_' + '{}.png'.format(self.args.seed))
        Image.fromarray(sample, 'RGB').save(figsave_path)

    def read_image(self, image_name):
        data = np.array(Image.open(image_name))
        data = np.expand_dims(data, 0)
        data = data.reshape(data.shape[0], -1)

        return data
