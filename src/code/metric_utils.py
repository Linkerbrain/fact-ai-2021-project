import torch
import torch.nn.functional as F

import numpy as np

import inversefed

class PrivacyMetrics:
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

    def _calculate_reconstruction_score(self, idx):
        """
        Calculates the reconstruction score
        """
        # load in dataset mean and std
        if self.dataset_name == 'cifar100':
            dm = torch.as_tensor(inversefed.consts.cifar10_mean, device=self.device)[:, None,
                                                                        None]
            ds = torch.as_tensor(inversefed.consts.cifar10_std, device=self.device)[:, None,
                                                                        None]
        elif self.dataset_name == 'FashionMinist':
            dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
            ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
        else:
            raise NotImplementedError

        # get labels/real image
        ground_truth, labels = [], []
        while len(labels) < self.num_images_per_recon_test:
            img, label = self.validloader.dataset[idx]
            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label, ), device=self.device))
                ground_truth.append(img.to(self.device))

        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)

        # prepare model
        self.model.eval()
        self.model.zero_grad()

        # calcuate ori dW
        target_loss, _, _ = self.loss_fn(self.model(ground_truth), labels)
        input_gradient = torch.autograd.grad(target_loss, self.model.parameters())

        metric = 'cos'

        # attack model
        self.model.eval()
        dw_list = list()
        dx_list = list()
        bin_num = 20
        noise_input = (torch.rand((ground_truth.shape)).cuda() - dm) / ds
        for dis_iter in range(bin_num + 1):
            self.model.zero_grad()
            fake_ground_truth = (1.0 / bin_num * dis_iter * ground_truth +
                                1. / bin_num *
                                (bin_num - dis_iter) * noise_input).detach()
            fake_dw = self._calculate_dw(self.model, fake_ground_truth, labels, self.loss_fn)
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

    def _calculate_accuracy_score(self, idx_list):
        """
        Calculate estimated accuracy score
        """
        # get labels/real image
        ground_truth, labels = [], []
        for idx in idx_list:
            img, label = self.validloader.dataset[idx]
            idx += 1
            if label not in labels:
                labels.append(torch.as_tensor((label, ), device=self.setup['device']))
                ground_truth.append(img.to(**self.setup))

        ground_truth = torch.stack(ground_truth)
        labels = torch.cat(labels)

        # prepare model
        self.model.zero_grad()

        # do the maths
        jacobs, labels = self._get_batch_jacobian(self.model, ground_truth, labels)
        jacobs = jacobs.reshape(jacobs.size(0), -1).cpu().numpy()

        return self._eval_score(jacobs, labels)