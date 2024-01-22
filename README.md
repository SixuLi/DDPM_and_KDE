# A Good Score Does not Lead to A Good Generative Model

We provide code for all the experiments presented in paper [A Good Score Does not Lead to A Good Generative Model](https://arxiv.org/abs/2401.04856).

The organization of code is as follows:
* Source code is present in `src` directory.
* Bash files required to run the experiments along with all the hyperparameters used are in the `bin` directory.

## General Requirements

The main dependencies for running the code are
* numpy
* matplotlib
* seaborn

## Running Experiments

We provide detailed instructions on running each experiment.

Each experiment has a bash command stored in the bash file "bin/run_simulations.sh" along with the hyperparameters and random seeds used in the experiment. Corresponding command in the relevant bash file needs to be uncommented before running the experiment. For most of the code, commands and argument names are self-explanatory.

### Synthetic Dataset

We run the experiments using 2-dimensional Gaussian as the target distribution. For all the experiments in this section, one only need to run the corresponding bash commands to reproduce the results.

### CIFAR10 Dataset

We run the experiments using CIFAR10 dataset.

#### Generate samples from Gaussian KDE:
Run the corresponding bash command.

#### Generate samples from DDPM with empirical optimal score function:
We follow the code [DiffMemorize](https://github.com/sail-sg/DiffMemorize) provided in paper [On Memorization in Diffusion Models](https://arxiv.org/abs/2310.02664).
* Step 1:
  Prepare the environments by running the following commands to install python libraries:
  
  ```
  pip install -r DiffMemorize/requirements.txt
  ```

* Step 2:
  Download CIFAR10 dataset and save it to `datasets/cifar10` by the following commands:
  
  ```
  mkdir datasets
  mkdir datasets/cifar10
  wget -P datasets/cifar10 https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
  ```
* Step 3:
  Prepare the full training dataset of CIFAR10:
  
  ```
  python DiffMemorize/dataset_tool.py --source=datasets/cifar10/cifar-10-python.tar.gz --dest=datasets/cifar10/cifar10-train.zip
  ```
* Step 4:  Run the corresponding bash command in `bin/run_simulations.sh`.
  


