#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
SRC='src'

##### Uncomment the block to run the code #####

##### Synthetic data distribution: 2d-Gaussian #####

#### Generate samples from DDPM with empirical optimal score function and true score function ####
#for seed in 0
#do
#  python "${SRC}"/main.py \
#    --experiment_name 'DDPM_generation_2d_gaussian' \
#    --T 5 \
#    --h 0.0005 \
#    --num_training_data 100 \
#    --num_generated_samples 1000 \
#    --seed "$seed"
#done

#### Estimate score approximation error ####
#for seed in 0
#do
#  python "${SRC}"/main.py \
#    --experiment_name 'Estimation_score_approximation_error' \
#    --T 5 \
#    --h 0.02 \
#    --seed "$seed"
#done


##### Real-world data distribution: CIFAR10 dataset #####

#### Generate samples from Gaussian KDE ####
for seed in {0..10}
do
  python "${SRC}"/main.py \
      --experiment_name "KDE_generation_cifar10" \
      --seed "$seed"
done


#### Generate samples from DDPM with empirical optimal score function ####







