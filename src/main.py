import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
import os
import scipy.stats as stats
import argparse
import logging
import init
import time
from algorithm import Generative_model
from KDE import KernelDensityEstimator
from evaluate_total_correlation import Estimate_TC

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
    parser.add_argument('--scaling_factor', type=float, default=0.1)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--d', type=int, default=2)
    parser.add_argument('--M', type=int, default=10)
    parser.add_argument('--K', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--cov', type=float, default=1.0)

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(args.experiment_name)
    logging.info("Seed {}".format(args.seed))
    # logging.info("gamma {}".format(args.gamma))

    train_init = init.Init(args=args)

    if args.experiment_name == 'KDE_generation_cifar10':
        model = KernelDensityEstimator(dataset_name='CIFAR10', args=args, train_init=train_init)
        original_sample, generate_sample = model.random_sample(scaling_factor=args.scaling_factor)
        model.visualization(sample=original_sample, tag='original')
        model.visualization(sample=generate_sample, tag='KDE_generate')

    elif args.experiment_name == 'estimate_total_correlation':
        starting_time = time.time()
        estimtated_tc_list = []
        for i in range(10):
            estimate_tc = Estimate_TC(args=args, train_init=train_init)
            estimated_tc = estimate_tc.MC_approx()
            estimtated_tc_list.append(estimated_tc)
            print('Estimated Total Correlation = ', estimated_tc)
        with open(os.path.join(train_init.output_path, 'result.txt'), 'a') as f:
            f.write("List of estimated total correlation for gamma={}: {}\n\n".format(args.gamma, estimtated_tc_list))
        ending_time = time.time()
        print('Total number time = ', ending_time - starting_time)

    else:
        model = Generative_model(args=args, train_init=train_init)

        if model.args.experiment_name == 'Estimation_score_approximation_error':
            num_training_data_list = np.array(range(100,2001,100))
            score_approximation_error_list = np.zeros_like(num_training_data_list, dtype=float)
            for i, num_training_data in enumerate(num_training_data_list):
                print('Number of samples = {}'.format(num_training_data))
                score_approximation_error_list[i] = model.score_approximation_error(num_x=1000, num_y=num_training_data,
                                                                                            rand_sample_time=10)
            print(score_approximation_error_list)
            model.visualization_score_approximation_error(num_training_data_list,score_approximation_error_list)

        elif model.args.experiment_name == 'DDPM_generation_2d_gaussian':
            model.run_generative_algorithm()
            model.visualization_sample_generation(tag='ddpm')
            model.visualization_sample_generation(tag='true_process')



