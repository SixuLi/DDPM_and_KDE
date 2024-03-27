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
for seed in 1
do
  python "${SRC}"/main.py \
      --experiment_name "KDE_generation_cifar10" \
      --scaling_factor 0.1 \
      --seed "$seed"
done


#### Generate samples from DDPM with empirical optimal score function ####
#python "${SRC}"/DiffMemorize/generate_optim.py \
#    --outdir=results/fid-tmp-optim-early-stop-2 \
#    --early_stop=2 \
#    --seeds=20-30 \
#    --subdirs \
#    --network=datasets/cifar10/cifar10-train.zip

##### Estimate Total Correlation for Gaussian KDE #####
#for gamma in 0.001 0.005 0.01 0.05 0.1 0.5 1.0 5.0 10.0 50.0
#do
#  python "${SRC}"/main.py \
#    --experiment_name "estimate_total_correlation" \
#    --N 100 \
#    --d 20 \
#    --M 10 \
#    --K 200 \
#    --gamma $gamma \
#    --cov 1 \
#    --seed 1
#done








