import os, sys

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

import random
import copy
import argparse
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm

from PIL import Image
import inversefed
from inversefed.data.data_processing import _build_cifar100, _get_meanstd

from args_utils import get_args
from data_utils import preprocess_data
from model_utils import create_model
from metric_utils import PrivacyMetrics

# set seed
seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)


class PolicyEvaluator(PrivacyMetrics):
    def __init__(self, args):
        """
        Evaluates a policy
        """
        # policy settings
        self.aug_list = args.aug_list

        # data settings
        self.dataset_name = args.dataset_name
        self.data_path = args.data_path
        self.transform_mode = args.transform_mode
        self.normalize = args.normalize
        self.augment_validation = args.augment_validation

        # model settings
        self.architecture = args.architecture
        self.device = args.device
        self.epochs = args.epochs # TODO: This does not really get used

        self.model_trained_partially_path = args.model_trained_partially_path

        # eval settings
        self.recon_tests_per_policy = args.recon_tests_per_policy
        self.num_images_per_recon_test = args.num_images_per_recon_test
        self.results_dir = args.results_dir

        # init InverseFed (The module that reconstructs the image from the gradient)
        self.setup = inversefed.utils.system_startup()
        defs = inversefed.training_strategy('conservative')
        defs.epochs = self.epochs

        # init data
        self.loss_fn, self.trainloader, self.validloader = preprocess_data(dataset_name=self.dataset_name, data_path=self.data_path, batch_size=defs.batch_size, transform_mode=self.transform_mode, aug_list=self.aug_list, normalize=self.normalize, augment_validation=self.augment_validation)

        # init model
        self.model = create_model(self.dataset_name, self.architecture)
        self.model.to(self.device)

        # save model status for future reference
        self.old_state_dict = copy.deepcopy(self.model.state_dict())

        # load model
        self.model.load_state_dict(
            torch.load(self.model_trained_partially_path))

    def evaluate_policy(self):
        self.model.eval()

        # evaluate reconstruction
        print(f"> Evaluating reconstruction metrics for {self.aug_list}...")

        sample_list = [200 + i * 5 for i in range(self.recon_tests_per_policy)] # WTF is this sampling ?? TODO: Fix this, it's from the original code

        recon_scores = []
        for sample_idx in sample_list:
            score = self._calculate_reconstruction_score(sample_idx)

            recon_scores.append(score)

        print(f"Done! {self.aug_list} scored {np.mean(recon_scores):.3} on average in reconstruction/area under curve metric")

        # save reconstruction to disk
        filefolder = 'reconstruction/data_{}_arch_{}/'.format(self.dataset_name, self.architecture)
        filename = self.aug_list
        root_dir = os.path.join(self.results_dir, filefolder)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        if len(recon_scores) > 0:
            np.save(os.path.join(root_dir, filename), recon_scores)

        # evaluate accuracy
        print(f"Evaluating accuracy metrics for {self.aug_list}...")

        # reset model
        self.model.load_state_dict(self.old_state_dict)

        acc_scores = []
        for run in range(10): # TODO fix these numbers
            large_sample_list = [200 + run * 100 + i for i in range(100)] # again they use very weird sampling

            score = self._calculate_accuracy_score(large_sample_list)
            acc_scores.append(score)

        print(f"Done! {self.aug_list} scored {np.mean(acc_scores)} on accuracy estimation based on jacobian metric")

        # save to disk
        filefolder = 'accuracy/data_{}_arch_{}/'.format(self.dataset_name, self.architecture)
        filename = self.aug_list
        root_dir = os.path.join(self.results_dir, filefolder)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        if len(acc_scores) > 0:
            np.save(os.path.join(root_dir, filename), acc_scores)

        print(f'> Finished evaluating {self.aug_list}', )


if __name__ == "__main__":
    args = get_args()

    evaluator = PolicyEvaluator(args
    )

    evaluator.evaluate_policy()
