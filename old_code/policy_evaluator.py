import os, sys

sys.path.insert(0, './')
import torch
import torchvision

seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random

random.seed(seed)

import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import inversefed
import torchvision.transforms as transforms
import argparse
from autoaugment import SubPolicy
from inversefed.data.data_processing import _build_cifar100, _get_meanstd
import torch.nn.functional as F
from benchmark.comm import create_model, build_transform, preprocess, create_config
import policy
import copy

from tqdm import tqdm

policies = policy.policies

class EvalSettings():
    """
    Quick fix so we don't have to edit preprocess file
    TODO: fix this
    """
    def __init__(self, mode, aug_list, rlabel, arch, data, epochs):
        self.mode = mode
        self.aug_list = aug_list
        self.rlabel = rlabel
        self.arch = arch
        self.data = data
        self.epochs = epochs

class PolicyEvaluator():
    def __init__(self, mode, aug_list, rlabel, arch, data, epochs, iters_per_policy):
        """
            opt = {
                "mode" : None,
                "aug_list" : None,
                "rlabel" : False,
                "arch" : None,
                "data" : None,
                "epochs" : None
            }
        """
        self.mode = mode
        self.aug_list = aug_list
        self.rlabel = rlabel
        self.arch = arch
        self.data = data
        self.epochs = epochs
        self.iters_per_policy = iters_per_policy

        print("DEBUG: Initiating policy evaluator with aug list:", aug_list)

        # init env
        self.setup = inversefed.utils.system_startup()
        defs = inversefed.training_strategy('conservative')
        defs.epochs = self.epochs

        # init training

        # QUICK FIX
        ugly_opt = EvalSettings(self.mode, self.aug_list, self.rlabel, self.arch, self.data, self.epochs)

        self.loss_fn, self.trainloader, self.validloader = preprocess(ugly_opt, defs, valid=True)

        assert mode in ['normal', 'aug', 'crop']

        self.num_images = 1

        self.model = create_model(ugly_opt)
        self.model.to(**self.setup)
        
        self.old_state_dict = copy.deepcopy(self.model.state_dict())
        self.model.load_state_dict(
            torch.load('checkpoints/tiny_data_{}_arch_{}/{}.pth'.format(
                self.data, self.arch, self.epochs)))

    def evaluate_policy(self):
        self.model.eval()
        metric_list = list()

        import time
        start = time.time()

        """
        Weird choice of sampling, no seed just hardcoded range?
        """
        sample_list = [200 + i * 5 for i in range(self.iters_per_policy)]
        metric_list = list()

        print(f"Evaluating search metrics for {self.aug_list}...")
        for attack_id, idx in enumerate(sample_list):
            metric = self._reconstruct(idx, self.model, self.loss_fn, self.trainloader, self.validloader)
            metric_list.append(metric)
            print(".", end="")
        print()

        pathname = 'search/data_{}_arch_{}/{}'.format(self.data, self.arch,
                                                    self.aug_list)
        root_dir = os.path.dirname(pathname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        if len(metric_list) > 0:
            print(np.mean(metric_list))
            np.save(pathname, metric_list)

        print(f"Done! {self.aug_list} scored {np.mean(metric_list):.2} mean on reconstruction/area under curve metric")

        # maybe need old_state_dict
        self.model.load_state_dict(self.old_state_dict)

        
        print(f"Evaluating accuracy metrics for {self.aug_list}...")
        score_list = list()
        for run in range(10):
            """
            Weird choice of sampling, no seed just hardcoded range?
            """
            large_sample_list = [200 + run * 100 + i for i in range(100)] # Random numbers ?
            score = self._accuracy_metric(large_sample_list, self.model, self.loss_fn, self.trainloader,
                                    self.validloader)
            score_list.append(score)

            print(".", end="")
        print()

        print(f"Done! {self.aug_list} scored {np.mean(score_list):3.2} mean on accuracy estimation based on jacobian metric")

        pathname = 'accuracy/data_{}_arch_{}/{}'.format(self.data, self.arch,
                                                        self.aug_list)
        root_dir = os.path.dirname(pathname)
        if not os.path.exists(root_dir):
            os.makedirs(root_dir)
        np.save(pathname, score_list)

        print(f'Finished evaluating {self.aug_list} in {time.time() - start} seconds', )

    def _eval_score(self, jacob, labels=None):
        corrs = np.corrcoef(jacob)
        v, _ = np.linalg.eig(corrs)
        k = 1e-5
        return -np.sum(np.log(v + k) + 1. / (v + k))

    def _get_batch_jacobian(self, net, x, target):
        net.eval()
        net.zero_grad()
        x.requires_grad_(True)
        y = net(x)
        y.backward(torch.ones_like(y))
        jacob = x.grad.detach()
        return jacob, target.detach()


    def _calculate_dw(self, model, inputs, labels, loss_fn):
        model.zero_grad()
        target_loss, _, _ = loss_fn(model(inputs), labels)
        dw = torch.autograd.grad(target_loss, model.parameters())
        return dw


    def _cal_dis(self, a, b, metric='L2'):
        a, b = a.flatten(), b.flatten()
        if metric == 'L2':
            return torch.mean((a - b) * (a - b)).item()
        elif metric == 'L1':
            return torch.mean(torch.abs(a - b)).item()
        elif metric == 'cos':
            return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item()
        else:
            raise NotImplementedError


    def _accuracy_metric(self, idx_list, model, loss_fn, trainloader, validloader):
        if self.data == 'cifar100':
            dm = torch.as_tensor(inversefed.consts.cifar10_mean, **self.setup)[:, None,
                                                                        None]
            ds = torch.as_tensor(inversefed.consts.cifar10_std, **self.setup)[:, None,
                                                                        None]
        elif self.data == 'FashionMinist':
            dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
            ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
        else:
            raise NotImplementedError

        # prepare data
        ground_truth, labels = [], []
        for idx in idx_list:
            img, label = validloader.dataset[idx]
            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label, ), device=self.setup['device']))
                ground_truth.append(img.to(**self.setup))

        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)
        model.zero_grad()
        jacobs, labels = self._get_batch_jacobian(model, ground_truth, labels)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()
        return self._eval_score(jacobs, labels)


    def _reconstruct(self, idx, model, loss_fn, trainloader, validloader):
        if self.data == 'cifar100':
            dm = torch.as_tensor(inversefed.consts.cifar10_mean, **self.setup)[:, None,
                                                                        None]
            ds = torch.as_tensor(inversefed.consts.cifar10_std, **self.setup)[:, None,
                                                                        None]
        elif self.data == 'FashionMinist':
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
                labels.append(torch.as_tensor((label, ), device=self.setup['device']))
                ground_truth.append(img.to(**self.setup))

        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)
        model.zero_grad()
        # calcuate ori dW
        target_loss, _, _ = loss_fn(model(ground_truth), labels)
        input_gradient = torch.autograd.grad(target_loss, model.parameters())

        metric = 'cos'

        # attack model
        model.eval()
        dw_list = list()
        dx_list = list()
        bin_num = 20
        noise_input = (torch.rand((ground_truth.shape)).cuda() - dm) / ds
        for dis_iter in range(bin_num + 1):
            model.zero_grad()
            fake_ground_truth = (1.0 / bin_num * dis_iter * ground_truth +
                                1. / bin_num *
                                (bin_num - dis_iter) * noise_input).detach()
            fake_dw = self._calculate_dw(model, fake_ground_truth, labels, loss_fn)
            dw_loss = sum([
                self._cal_dis(dw_a, dw_b, metric=metric)
                for dw_a, dw_b in zip(fake_dw, input_gradient)
            ]) / len(input_gradient)

            dw_list.append(dw_loss)

        interval_distance = self._cal_dis(noise_input, ground_truth,
                                    metric='L1') / bin_num

        def area_ratio(y_list, inter):
            area = 0
            max_area = inter * bin_num
            for idx in range(1, len(y_list)):
                prev = y_list[idx - 1]
                cur = y_list[idx]
                area += (prev + cur) * inter / 2
            return area / max_area

        return area_ratio(dw_list, interval_distance)

if __name__ == '__main__':
    evaluator = PolicyEvaluator(mode="aug", aug_list="1-2-3", rlabel=False, arch="ResNet20-4", data="cifar100", epochs=100, iters_per_policy=100)

    evaluator.evaluate_policy()
