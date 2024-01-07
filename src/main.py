import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
import argparse
import logging
import init
from algorithm import Generative_model
from KDE import KernelDensityEstimator

sns.set_style("darkgrid", {'grid.linestyle': '--'})
sns.set_context('poster')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, default='test')
    parser.add_argument('--result_path', type=str, default='./results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--T', type=int, default=1)
    parser.add_argument('--h', type=float, default=0.0005)
    parser.add_argument('--num_training_data', type=int, default=10)
    parser.add_argument('--num_generated_samples', type=int, default=20)
    parser.add_argument('--is_early_stop', default=False, action='store_true')
    parser.add_argument('--is_explicit_sample', default=False, action='store_true')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(args.experiment_name)
    logging.info("Seed {}".format(args.seed))

    train_init = init.Init(args=args)

    if args.experiment_name == 'KDE_generation_cifar10':
        model = KernelDensityEstimator(dataset_name='CIFAR10', args=args, train_init=train_init)
        if model.args.is_explicit_sample:
            for i in range(8):
                model.args.seed = i
                sample = model.read_image('./results/fid-tmp-optim-early-stop-0/000000/00002{}.png'.format(i))
                original_sample, generate_sample = model.explicit_sample(scaling_factor=0.01, data=sample[0])
                model.visualization(sample=original_sample, tag='original')
                model.visualization(sample=generate_sample, tag='KDE_generate')
        else:
            original_sample, generate_sample = model.random_sample(scaling_factor=0.1)
            model.visualization(sample=original_sample, tag='original')
            model.visualization(sample=generate_sample, tag='KDE_generate')

    else:
        model = Generative_model(args=args, train_init=train_init)

        if model.args.experiment_name == 'Estimation_score_approximation_error':
            num_training_data_list = np.array(range(100,101,100))
            score_approximation_error_list = np.zeros_like(num_training_data_list, dtype=float)
            for i, num_training_data in enumerate(num_training_data_list):
                print('Number of samples = {}'.format(num_training_data))
                score_approximation_error_list[i] = model.score_approximation_error(num_x=1000, num_y=num_training_data,
                                                                                            rand_sample_time=10)
            print(score_approximation_error_list)
            # score_approximation_error_list = np.array([3.03330265, 1.57131465, 1.07974174, 0.82257152, 0.65978504, 0.57862584,
            #                                            0.50609356, 0.4406535,  0.39416571, 0.36625418, 0.34241554, 0.31937641,
            #                                            0.30400835, 0.28355453, 0.27308075, 0.23998752, 0.2215496,  0.20494394,
            #                                            0.19673546, 0.18940907])
            model.visualization_score_approximation_error(num_training_data_list,score_approximation_error_list)

        elif model.args.experiment_name == 'DDPM_generation_2d_gaussian':
            model.run_generative_algorithm()
            model.visualization_sample_generation(tag='ddpm')
            model.visualization_sample_generation(tag='true_process')



