#!/bin/bash


SBATCH –partition=v100
# partition v100 means: send the Job to the GPU Server Node
SBATCH –gres=gpu:1
# asks for one out of 4 GPUs
SBATCH --output=/your-path-to-working-directory/ML-Projectname_%j.out
# File to which STDOUT is written
SBATCH --error=/your-path-to-working-directory/ML-Projectname_%j.err
# File to which STDERR is written
SBATCH –mail-type=END,FAIL Type of email notification- BEGIN,END,FAIL,ALL
SBATCH --mail-user=yourmailaddress@avl.com # to whom mail will be sent