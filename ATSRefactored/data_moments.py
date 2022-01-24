import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm

import inversefed
from args_utils import parse_single_policy, parse_moment_matching
from data_utils import preprocess_data


def get_save_path(args):
    save_path = os.path.join(args.results_dir,
                             f"moments_{args.dataset_name}_mode_{args.transform_mode}_auglist_{args.aug_list}.pt")
    return save_path


def get_moments(args):
    save_path = get_save_path(args)
    return torch.load(save_path)


def calculate_moments(x, moment_range):
    moments = []
    mean = torch.mean(x, dim=(2, 3), keepdim=True)
    var = torch.mean((x - mean) ** 2, dim=(2, 3), keepdim=True)
    moments.append(mean.squeeze())
    moments.append(var.squeeze())
    std = torch.sqrt(var)
    for moment in range(3, moment_range):
        m = torch.mean((x - mean) ** moment, dim=(2, 3), keepdim=True) / std ** moment
        # Happens when std is 0, ideally we would like to ignore this moment then but this should be fine
        m[torch.isnan(m)] = 0
        moments.append(m.squeeze())
    return moments


def main(args):
    if args.transform_mode == 'crop':
        args.aug_list = ""
    print(args)
    # Load augmented data, fully processed
    defs = inversefed.training_strategy('conservative')
    loss_fn, trainset, validset = preprocess_data(dataset_name=args.dataset_name, data_path=args.data_path,
                                                  batch_size=defs.batch_size,
                                                  transform_mode=args.transform_mode, aug_list=args.aug_list,
                                                  normalize=args.normalize,
                                                  augment_validation=args.augment_validation)
    # dataloader = torch.utils.data.DataLoader(trainset, batch_size=defs.batch_size, shuffleFalse)
    moment_range = args.moment_range
    assert moment_range >= 3, "moment_range must be at least 3, which gives mean and variance"
    moments = list(torch.zeros(3) for _ in range(moment_range - 1))
    count = 0
    # Calculate image moments
    for x, y in tqdm(trainset):
        for i, moment in enumerate(calculate_moments(x, moment_range)):
            moments[i] += torch.mean(moment, dim=(0))
        count += 1
    moments = torch.stack([m / count for m in moments])

    save_path = get_save_path(args)
    print(f"Moments {moments} saved to {save_path}")

    torch.save(moments, save_path)


if __name__ == "__main__":
    parser = ArgumentParser(description="Moment matching")
    parse_single_policy(parser)
    parse_moment_matching(parser)
    args = parser.parse_args()
    main(args)
