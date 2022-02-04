import torch
import torchvision
import torchvision.transforms as transforms

import inversefed
from inversefed.data.loss import Classification
from policy import Policy


def _parse_aug_list(aug_list):
    if '+' not in aug_list:
        return [int(idx) for idx in aug_list.split('-')]
    else:
        ret_list = list()
        for aug in aug_list.split('+'):
            ret_list.append([int(idx) for idx in aug.split('-')])
        return ret_list


def preprocess_data(dataset_name, data_path, batch_size, transform_mode, aug_list, normalize, augment_validation):
    """
    Loads the data, creates a dataloader which applies the transformations

    The transformation consist of both Global Preprocess transformations like normalisation
    and the augmentations that are defined
    """
    # load data
    if dataset_name == 'cifar100':
        loss_fn, trainset, validset = _data_cifar100(data_path)
    elif dataset_name == 'FashionMnist':
        loss_fn, trainset, validset = _data_fashionmninst(data_path)
    else:
        raise NotImplementedError()

    # define transformations
    if transform_mode == 'aug':
        augmentations = _parse_aug_list(aug_list)
    else:
        augmentations = []
    transformations = make_transformations(dataset_name, transform_mode, augmentations, normalize)

    trainset.transform = transformations

    if augment_validation:
        validset.transform = transformations

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    return loss_fn, trainloader, validloader


def _data_cifar100(data_path):
    loss_fn = Classification()

    trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=True,
                                             transform=transforms.ToTensor())

    return loss_fn, trainset, validset


def _data_fashionmninst(data_path):
    loss_fn = Classification()

    trainset = torchvision.datasets.FashionMNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
    validset = torchvision.datasets.FashionMNIST(data_path, train=False, download=True, transform=transforms.ToTensor())

    return loss_fn, trainset, validset


def make_transformations(dataset_name, mode, augmentations, normalize):
    """
    Builds transformation list
    consisting of preprocessing (data dependent)
    and the augmentations
    """

    # settings based on dataset
    if dataset_name == 'cifar100':
        data_mean, data_std = inversefed.consts.cifar100_mean, inversefed.consts.cifar100_std
    elif dataset_name == 'FashionMnist':
        data_mean, data_std = (0.1307,), (0.3081,)
    else:
        raise NotImplementedError

    transform_list = []
    # build transform list
    if mode != 'normal':
        transform_list += [transforms.RandomCrop(32, padding=4),
                           transforms.RandomHorizontalFlip()]
    if mode == 'aug' and len(augmentations) > 0:
        transform_list.append(Policy(augmentations))

    # add transforms for some datasets
    if dataset_name == 'FashionMinist':
        transform_list = [lambda x: transforms.functional.to_grayscale(x, num_output_channels=3)] + transform_list
        transform_list.append(lambda x: transforms.functional.to_grayscale(x, num_output_channels=1))
        transform_list.append(transforms.Resize(32))

    # These are the actual means and stds calculated from the training set,
    # which differ some from the consts given by inversefed, but in the interest
    # of keeping the setting similar to the original paper, we use the consts from inversefed
    # data_mean, data_std = [ 0.4382,  0.4178,  0.3772], list(np.sqrt([0.0700,  0.0657,  0.0639]))
    # add normalisation
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(data_mean, data_std) if normalize else transforms.Lambda(lambda x: x),
    ])

    transform = transforms.Compose(transform_list)
    return transform
