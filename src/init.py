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
        self.output_path = os.path.join(args.result_path, args.experiment_name + '_d_{}'.format(args.d))

        if args.seed == 0:
            make_dirs(self.result_path)
            make_dirs(self.output_path)
            args_state = {k: v for k, v in args._get_kwargs()}
            with open(os.path.join(self.output_path, 'result.txt'), 'w') as f:
                print(args_state, file=f)