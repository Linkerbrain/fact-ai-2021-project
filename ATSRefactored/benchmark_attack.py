import os
import sys

import stn
from args_utils import parse_attack, parse_moment_matching, parse_single_policy
from data_moments import calculate_moments

sys.path.insert(0, './')
import torch
import torchvision

seed = 23333
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import random

random.seed(seed)

import numpy as np
import inversefed
import argparse
# from benchmark.comm import create_model, preprocess, create_config
from data_utils import preprocess_data
from model_utils import create_model

# Stick to inversed optim mode
config = dict(signed=True,
              boxed=True,
              cost_fn='sim',
              indices='def',
              weights='equal',
              lr=0.1,
              optim='adam',
              restarts=1,
              max_iterations=4800,
              # max_iterations=1,
              total_variation=1e-4,
              init='randn',
              filter='none',
              lr_decay=True,
              scoring_choice='loss')

reaugment_modes = dict(
    none=dict(
        prep_with='none',
    ),
    # affine=dict(
    #     prep_module=stn.AffineTransform(0.1)
    # ),
    translate_clipped1=dict(
        prep_with='translate',
        prep_args=dict(clipX=1.1, clipY=0.2)
    ),
    shiftL=dict(
        prep_with='shiftL',
    ),
    shiftR=dict(
        prep_with='shiftR',
    )
)


def moment_matching_loss(moments, image, moment_range, alpha=1):
    m = torch.stack(calculate_moments(image, moment_range)).squeeze()
    return moments[:moment_range].sub(m).pow(2).sum() * alpha


def create_save_dir(args):
    return 'results/images/data_{}_arch_{}_epoch_{}_optim_{}_mode_{}_auglist_{}_rlabel_{}_reaugment_{}'.format(
        args.dataset_name,
        args.architecture,
        args.epochs,
        args.optim,
        args.transform_mode,
        args.aug_list,
        args.rlabel,
        args.reaugment,
        # args.moment_matching,
        # args.affine_transform,
        # args.translation_transform,
        # 'left' if args.shift_left else 'right' if args.shift_right else 'none'
        # , str(args.translation_clip) + str(args.shear_clip) + str(args.scale_clip))
    )


def reconstruct(args, setup, config, idx, model, loss_fn, trainloader, validloader):
    if args.dataset_name == 'cifar100':
        dm = torch.as_tensor(inversefed.consts.cifar10_mean, **setup)[:, None, None]
        ds = torch.as_tensor(inversefed.consts.cifar10_std, **setup)[:, None, None]
    elif args.dataset_name == 'FashionMinist':
        dm = torch.Tensor([0.1307]).view(1, 1, 1).cuda()
        ds = torch.Tensor([0.3081]).view(1, 1, 1).cuda()
    else:
        raise NotImplementedError

    # prepare data
    ground_truth, labels = [], []
    while len(labels) < args.num_images:
        img, label = validloader.dataset[idx]
        idx += 1
        if label not in labels:
            labels.append(torch.as_tensor((label,), device=setup['device']))
            ground_truth.append(img.to(**setup))

    ground_truth = torch.stack(ground_truth)
    labels = torch.cat(labels)
    model.zero_grad()
    target_loss, _, _ = loss_fn(model(ground_truth), labels)
    param_list = [param for param in model.parameters() if param.requires_grad]
    input_gradient = torch.autograd.grad(target_loss, param_list)

    # ground_truth_moments = get_moments(args).cuda()

    reaugment_mode = reaugment_modes[args.reaugment]

    # attack
    print('ground truth label is ', labels)
    extra_loss = lambda image: 0.0
    # if args.moment_matching:
    #     extra_loss = lambda image: moment_matching_loss(ground_truth_moments, image, args.moment_range)
    prep_module = None
    if reaugment_mode['prep_with'] == 'affine':
        prep_module = stn.AffineTransform().to(args.device)
    elif reaugment_mode['prep_with'] == 'translate':
        prep_module = stn.Translation(**reaugment_mode['prep_args']).to(args.device)
    elif reaugment_mode['prep_with'] in ['shiftL', 'shiftR']:
        prep_module = stn.Translation().to(args.device)
        prep_module.xy.requires_grad = False
        prep_module.xy[0, 0] = 1 if reaugment_mode['prep_with'] == 'shiftL' else -1

    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=args.num_images,
                                                   extra_loss_fn=extra_loss, preprocessing_module=prep_module)
    if args.dataset_name == 'cifar100':
        shape = (3, 32, 32)
    elif args.dataset_name == 'FashionMinist':
        shape = (1, 32, 32)

    if args.rlabel:
        output, stats = rec_machine.reconstruct(input_gradient, None, img_shape=shape)  # reconstruction label
    else:
        output, stats = rec_machine.reconstruct(input_gradient, labels, img_shape=shape)  # specify label

    output_denormalized = (output * ds + dm).squeeze()
    input_denormalized = (ground_truth * ds + dm).cpu().squeeze()

    output_transformed = (
        prep_module(output_denormalized.unsqueeze(0)) if prep_module is not None else output).cpu().squeeze()

    # Note on code below: x and y are flipped, but the code works as long as the flip is consistent

    print(shape, output_transformed.shape, input_denormalized.shape)
    # Align both such that the black regions match, since it is not a requirement for the model to predict where the black regions are, and it usually fails to do so. Essentially, we align the images to the top left corner.
    output_mask = torch.where(output_transformed.sum(axis=0) != 0)
    output_isolated = torch.zeros(shape)
    cx, cy = output_mask[0][0], output_mask[1][0]
    print(f"Output shift: {cx}, {cy}")
    output_isolated[:, :shape[1] - cx, :shape[2] - cy] = output_transformed[:,
                                                         cx:, cy:]

    # Most common value will be the black tint in the case of the 3-1-7 transform
    black = torch.mode(input_denormalized.abs().sum(axis=0).flatten())[0]
    # Add a leeway term because the borders have a bit of a blur
    input_mask = torch.where(input_denormalized.abs().sum(axis=0) > black + 0.1)
    input_isolated = torch.zeros(shape)
    cx, cy = input_mask[0][0], input_mask[1][0]
    print(f"Input shift: {cx}, {cy}")
    input_isolated[:, :shape[1] - cx, :shape[2] - cy] = input_denormalized[:, cx:,
                                                        cy:]

    # Align the output to the input such that the MSE is minimal
    # Simple brute force as the images are small enough, if they are not
    # one could use feature matching or something else.
    ty, tx = prep_module.xy.detach().cpu().squeeze() if reaugment_mode['prep_with'] == 'translate' else (0, 0)
    tx = int(-tx / 2 * shape[1])
    ty = int(-ty / 2 * shape[2])
    print(f"Translated by y={tx}, x={ty}")
    # output_aligned = None
    # areaX, areaY = None, None
    # bcx, bcy = None, None
    # best_mse = float('inf')
    # for cx in range(0, abs(tx)+1) if tx < 0 else range(-abs(tx), 1): #range(0 if tx < 0 else -tx, tx+1 if tx < 0 else 1):
    #     for cy in range(0, abs(ty)+1) if ty < 0 else range(-abs(ty), 1): #range(0 if ty < 0 else -ty, ty+1 if ty < 0 else 1):
    #         xy = -torch.tensor([[cx * 2 / shape[1], cy * 2 / shape[2]]], device=args.device, dtype=torch.float)
    #         translated = stn.translate(output_transformed.unsqueeze(0), xy).squeeze().cpu()
    #         mse = ((translated - input_denormalized) ** 2).mean()
    #         if mse < best_mse:
    #             output_aligned = translated
    #             best_mse = mse
    #             areaX = (tx + cx, shape[1]) if cx > 0 else (tx - cx, shape[1] - cx)
    #             areaY = (ty + cy, shape[2]) if cy > 0 else (ty - cy, shape[2] - cy)
    #             bcx, bcy = cx, cy
    # print(f"Aligned with y={bcx}, x={bcy}")
    # print(f"Area: {areaX}, {areaY}")

    mean_loss = torch.mean((input_isolated - output_isolated) * (input_isolated - output_isolated))
    print("after optimization, the true mse loss {}".format(mean_loss))

    save_dir = create_save_dir(args)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torchvision.utils.save_image(output_denormalized.cpu().clone(), '{}/rec_untransformed_{}.jpg'.format(save_dir, idx))
    torchvision.utils.save_image(output_transformed.cpu().clone(), '{}/rec_{}.jpg'.format(save_dir, idx))
    torchvision.utils.save_image(output_isolated.cpu().clone(), '{}/rec_isolated_{}.jpg'.format(save_dir, idx))
    # torchvision.utils.save_image(output_aligned.cpu().clone(), '{}/rec_aligned_{}.jpg'.format(save_dir, idx))
    torchvision.utils.save_image(input_denormalized.cpu().clone(), '{}/ori_{}.jpg'.format(save_dir, idx))
    torchvision.utils.save_image(input_isolated.cpu().clone(), '{}/ori_isolated_{}.jpg'.format(save_dir, idx))

    feat_mse = (model(prep_module(output).detach()) - model(ground_truth)).pow(2).mean()
    test_mse = (output_transformed.detach() - input_denormalized).pow(2).mean().cpu().detach().numpy()
    test_psnr = inversefed.metrics.psnr(output_transformed.unsqueeze(0), input_denormalized.unsqueeze(0))
    test_mse_untransformed = (output_denormalized.cpu().detach() - input_denormalized).pow(
        2).mean().cpu().detach().numpy()
    test_psnr_untransformed = inversefed.metrics.psnr(output_denormalized.cpu().unsqueeze(0),
                                                      input_denormalized.unsqueeze(0))
    test_mse_isolated = (output_isolated.detach() - input_isolated).pow(2).mean().cpu().detach().numpy()
    test_psnr_isolated = inversefed.metrics.psnr(output_isolated.unsqueeze(0), input_isolated.unsqueeze(0))
    test_mse_isolated_area = (output_isolated[:, :shape[1]-abs(tx), :shape[2]-abs(ty)].detach() - input_isolated[:, :shape[1]-abs(tx), :shape[2]-abs(ty)]).pow(2).mean().cpu().detach().numpy()
    test_psnr_isolated_area = inversefed.metrics.psnr(output_isolated[:, :shape[1]-abs(tx), :shape[2]-abs(ty)].unsqueeze(0), input_isolated[:, :shape[1]-abs(tx), :shape[2]-abs(ty)].unsqueeze(0))
    # test_mse_aligned = (output_aligned.detach() - input_denormalized).pow(2).mean().cpu().detach().numpy()
    # test_psnr_aligned = inversefed.metrics.psnr(output_aligned.unsqueeze(0), input_denormalized.unsqueeze(0))
    # test_mse_aligned_area = (
    #         output_aligned[:, areaX[0]:areaX[1], areaY[0]:areaY[1]].detach() - input_denormalized[:, areaX[0]:areaX[1],
    #                                                                         areaY[0]:areaY[1]]).pow(
    #     2).mean().cpu().detach().numpy()
    # test_psnr_aligned_area = inversefed.metrics.psnr(output_aligned[:, areaX[0]:areaX[1], areaY[0]:areaY[1]].unsqueeze(0),
    #                                                  input_denormalized[:, areaX[0]:areaX[1], areaY[0]:areaY[1]].unsqueeze(
    #                                                      0))

    return {
        'feat_mse': feat_mse,
        'test_mse': test_mse,
        'test_psnr': test_psnr,
        'test_mse_untransformed': test_mse_untransformed,
        'test_psnr_untransformed': test_psnr_untransformed,
        'test_mse_isolated': test_mse_isolated,
        'test_psnr_isolated': test_psnr_isolated,
        'test_mse_isolated_area': test_mse_isolated_area,
        'test_psnr_isolated_area': test_psnr_isolated_area,
    #     'test_mse_aligned': test_mse_aligned,
    #     'test_psnr_aligned': test_psnr_aligned,
    #     'test_mse_aligned_area': test_mse_aligned_area,
    #     'test_psnr_aligned_area': test_psnr_aligned_area,
    }


def create_checkpoint_dir(args):
    return 'checkpoints/data_{}_arch_{}_mode_{}_auglist_{}_rlabel_{}'.format(args.dataset_name, args.architecture,
                                                                             args.transform_mode,
                                                                             args.aug_list,
                                                                             args.rlabel)


def main(args, setup, config, defs):
    global trained_model
    print(args)
    loss_fn, trainloader, validloader = preprocess_data(dataset_name=args.dataset_name, data_path=args.data_path,
                                                        batch_size=defs.batch_size,
                                                        transform_mode=args.transform_mode, aug_list=args.aug_list,
                                                        normalize=args.normalize,
                                                        augment_validation=args.augment_validation)
    model = create_model(args.dataset_name, args.architecture)
    model.to(**setup)
    if args.epochs == 0:
        trained_model = False

    if trained_model:
        checkpoint_dir = create_checkpoint_dir(args)
        if 'normal' in checkpoint_dir:
            checkpoint_dir = checkpoint_dir.replace('normal', 'crop')
        filename = os.path.join(checkpoint_dir, str(defs.epochs) + '.pth')

        if not os.path.exists(filename):
            filename = os.path.join(checkpoint_dir, str(defs.epochs - 1) + '.pth')

        print(filename)
        assert os.path.exists(filename)
        model.load_state_dict(torch.load(filename))

    if args.rlabel:
        for name, param in model.named_parameters():
            if 'fc' in name:
                param.requires_grad = False

    save_dir = create_save_dir(args)

    model.eval()
    sample_list = [i for i in range(100)]
    # sample_list = [1, 25, 73, 86, 91]
    # sample_list = [9, 12, 30, 31, 77, 86, 90]
    metric_list = list()
    mse_loss = 0
    for attack_id, idx in enumerate(sample_list):
        if idx < args.resume:
            continue
        print('attach {}th in {}'.format(idx, args.aug_list))
        metric = reconstruct(args, setup, config, idx, model, loss_fn, trainloader, validloader)
        print(metric)
        metric_list.append(metric)
        np.save('{}/metric.npy'.format(save_dir), metric_list)
        print(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Reconstruct some image from a trained model.')
    parse_single_policy(parser)
    parse_attack(parser)
    parse_moment_matching(parser)

    args = parser.parse_args()
    args.optim = 'inversed'

    # init env
    setup = inversefed.utils.system_startup()
    defs = inversefed.training_strategy('conservative')
    defs.epochs = args.epochs

    # init training
    arch = args.architecture
    trained_model = True
    mode = args.transform_mode
    assert mode in ['normal', 'aug', 'crop']

    main(args, setup, config, defs)
