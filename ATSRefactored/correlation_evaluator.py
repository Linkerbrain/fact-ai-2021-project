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
from config_utils import create_config
from data_utils import preprocess_data
from model_utils import create_model
from metric_utils import PrivacyMetrics

import time

# set seed
seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)

class CorrelationEvaluator(PrivacyMetrics):
    def __init__(self, args):
        """
        Evaluates the privacy metric and attack PSNR of a policy
        """
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

        # attack settings
        self.num_images_to_evaluate = args.num_images_to_evaluate
        self.num_images = args.num_images
        self.rlabel = args.rlabel

        self.model_trained_partially_path = args.model_trained_partially_path

        # eval settings
        self.recon_tests_per_policy = args.recon_tests_per_policy
        self.num_images_per_recon_test = args.num_images_per_recon_test
        
        self.results_dir = args.results_dir

        # init InverseFed (The module that reconstructs the image from the gradient)
        self.setup = inversefed.utils.system_startup()
        defs = inversefed.training_strategy('conservative')
        defs.epochs = self.epochs

        # init attack config
        self.attack_type = args.attack_type
        self.max_iters = args.max_iters
        self.config = create_config(self.attack_type, self.max_iters)

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


    def _privacy_metric_eval(self):
        self.model.eval()

        # evaluate reconstruction
        print(f"> Evaluating privacy metrics for {self.aug_list}...")

        sample_list = [200 + i * 5 for i in range(self.recon_tests_per_policy)] # WTF is this sampling ?? TODO: Fix this, it's from the original code

        recon_scores = []
        for sample_idx in sample_list:
            score = self._calculate_reconstruction_score(sample_idx)
            recon_scores.append(score)

        print(f"Done! {self.aug_list} scored {np.mean(recon_scores):.3} on average in reconstruction/area under curve metric")

        return np.mean(recon_scores)

    def _gradient_reconstruct(self, idx, model, loss_fn, trainloader, validloader):
        if self.dataset_name == 'cifar100':
            dm = torch.as_tensor(inversefed.consts.cifar10_mean, device=self.device)[:, None, None]
            ds = torch.as_tensor(inversefed.consts.cifar10_std, device=self.device)[:, None, None]
        elif self.dataset_name == 'FashionMinist':
            dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
            ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
        else:
            raise NotImplementedError

        # prepare data
        ground_truth, labels = [], []
        while len(labels) < self.num_images:
            img, label = validloader.dataset[idx]
            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label,), device=self.device))
                ground_truth.append(img.to(device=self.device))

        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)
        model.zero_grad()
        target_loss, _, _ = loss_fn(model(ground_truth), labels)
        param_list = [param for param in model.parameters() if param.requires_grad]
        input_gradient = torch.autograd.grad(target_loss, param_list)

        # attack
        print('ground truth label is ', labels)
        rec_machine = inversefed.GradientReconstructor(model, (dm, ds), self.config, num_images=self.num_images)
        if self.dataset_name == 'cifar100':
            shape = (3, 32, 32)
        elif self.dataset_namea == 'FashionMinist':
            shape = (1, 32, 32)

        if self.rlabel:
            output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=shape) # reconstruction label
        else:
            output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape) # specify label

        output_denormalized = output * ds + dm
        input_denormalized = ground_truth * ds + dm
        mean_loss = torch.mean((input_denormalized - output_denormalized) * (input_denormalized - output_denormalized))
        print("after optimization, the true mse loss {}".format(mean_loss))

        test_mse = (output_denormalized.detach() - input_denormalized).pow(2).mean().cpu().detach().numpy()
        feat_mse = (model(output.detach())- model(ground_truth)).pow(2).mean()
        test_psnr = inversefed.metrics.psnr(output_denormalized, input_denormalized)

        dic = {
            'test_mse': test_mse,
            'feat_mse': feat_mse,
            'test_psnr': test_psnr
        }

        return dic


    def _attack_PSNR_eval(self):
        self.model.eval()

        sample_list = [i for i in range(self.num_images_to_evaluate)]
        metric_list = list()
        mse_loss = 0
        for attack_id, idx in enumerate(sample_list):
            print('attack {}th in {}'.format(idx, self.aug_list))
            metric = self._gradient_reconstruct(idx, self.model, self.loss_fn, self.trainloader, self.validloader)
            metric_list.append(metric['test_psnr'])

        attack_PSNR = np.mean(metric_list)

        return attack_PSNR

    def evaluate_correlation(self):
        start_time = time.time()
        attack_PSNR = self._attack_PSNR_eval()
        privacy_score = self._privacy_metric_eval()
        end_time = time.time()

        print(f"\n{self.aug_list} scored {privacy_score} & {attack_PSNR} db (evalled in {end_time-start_time} seconds)\n")

        # save to disk
        filefolder = 'correlation/data_{}_arch_{}/'.format(self.dataset_name, self.architecture)
        root_dir = os.path.join(self.results_dir, filefolder)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)

        filename = self.aug_list + "_privacy_score"
        np.save(os.path.join(root_dir, filename), privacy_score)

        print(f"Saved privacy score in {os.path.join(root_dir, filename)}")

        filename = self.aug_list + "_attack_PSNR"
        np.save(os.path.join(root_dir, filename), attack_PSNR)
        
        print(f"Saved attack_PSNR in {os.path.join(root_dir, filename)}")


if __name__ == "__main__":
    args = get_args()

    evaluator = CorrelationEvaluator(args)

    evaluator.evaluate_correlation()