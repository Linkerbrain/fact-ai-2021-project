This is a reproduction study of the paper `Privacy-preserving Collaborative Learning with Automatic Transformation Search` by Wei Gao et al.

The original paper can be found here: https://arxiv.org/pdf/2011.12505.pdf

The original code can be found here: https://github.com/gaow0007/ATSPrivacy


This repository contains both the original code as well as a refactored version. There are notebooks present to reproduce the experiments from our study.

## Reproduction of Reproduction

This codebase is meant to be ran on a Linux system with a gpu (cuda). In the root folder an `environment.yml` anaconda environment is provided with the exact packages that this project requires.

The experiment that verifies the claim of the correlation between the privacy score and reconstruction PSNR can be found in the folder ATSRefactored under the name ```Result-Reproduction Correlation and Diversity.ipynb```. This notebook also contains the experiment for the second insight, which investigates the diversity of well-performing policies.

The experiment that verifies the reconstruction attack PSNR and accuracy of a trained model can be found in the root folder under the name ```Result-Reproduction Cifar100 Trained Models.ipynb```. This file contains the functions to show pre-run results, training the network for a single epoch and running the entire experiment. Additionally ```Result-Reproduction FMNIST Trained Models.ipynb``` contains the code to run the entire experiment for the FMNIST dataset.

## Folder structure

### ATSPrivacy-Framework

Old code from the original paper with slight edits in order to reproduce the results on our systems.

### ATSDatasets

ATSPrivacy-Framework but edited to work for F-Mnist

### ATSRefactored

New, refactored code, used for all our original experiments in the report. Symlinks in some places to the old code, in particular inversefed and the checkpoints is the same for both.
