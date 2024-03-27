import numpy as np
import os
import logging
import shutil


def make_dirs(dirname, rm=False):
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    elif rm:
        logging.info('Rm and mkdir {}'.format(dirname))
        shutil.rmtree(dirname)
        os.makedirs(dirname)

class Init:
    def __init__(self, args):
        np.random.seed(args.seed)
        self.result_path = args.result_path
        if args.experiment_name == 'KDE_generation_cifar10':
            self.output_path = os.path.join(args.result_path,
                                            args.experiment_name + '_scaling_factor_{}'.format(args.scaling_factor))
        elif args.experiment_name == 'estimate_total_correlation':
            self.output_path = os.path.join(args.result_path,
                                            args.experiment_name + '_d_{}'.format(args.d))
        else:
            self.output_path = os.path.join(args.result_path, args.experiment_name)

        if args.seed == 0:
            make_dirs(self.result_path)
            make_dirs(self.output_path)
            args_state = {k: v for k, v in args._get_kwargs()}
            with open(os.path.join(self.output_path, 'result.txt'), 'w') as f:
                print(args_state, file=f)