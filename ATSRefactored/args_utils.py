import argparse
from xmlrpc.client import Boolean


def parse_search_args(parser):
    # POLICY SEARCH SETTINGS
    parser.add_argument('--check_if_exists', type=bool, default=False)
    parser.add_argument('--num_random_policies_to_test', type=int, default=100)

    parser.add_argument('--augmentations_per_policy', type=int, default=3)
    parser.add_argument('--num_possible_augmentations', type=int, default=50)


def parse_single_policy(parser):
    # SINGLE POLICY SETTINGS
    parser.add_argument('--aug_list', type=str, default='45-36-28')

    parser.add_argument('--dataset_name', type=str, default='cifar100')
    parser.add_argument('--data_path', type=str, default='./data')

    parser.add_argument('--transform_mode', type=str, default='aug')
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--augment_validation', type=Boolean, default=True)

    parser.add_argument('--architecture', type=str, default='ResNet20-4')
    parser.add_argument('--model_trained_partially_path', type=str,
                        default='./models/tiny_data_cifar100_ResNet20-4_100epochs.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--recon_tests_per_policy', type=int, default=100)
    parser.add_argument('--num_images_per_recon_test', type=int, default=1)

    parser.add_argument('--results_dir', type=str, default='./results')


def parse_moment_matching(parser):
    parser.add_argument('--moment_range', type=int, default=5)


def parse_attack(parser):
    parser.add_argument('--attack_type', default="inversed", required=False, type=str, help='Attack config')
    parser.add_argument('--max_iters', default=2500, required=False, type=int, help='Number of iterations in attack')
    parser.add_argument('--num_images_to_evaluate', default=20, required=False, type=int, help='Number of samples of training set to evaluate in attack')
    # parser.add_argument('--aug_list', default=None, required=True, type=str, help='Vision model.')
    # parser.add_argument('--optim', default=None, required=True, type=str, help='Vision model.')
    # parser.add_argument('--mode', default=None, required=True, type=str, help='Mode.')
    parser.add_argument('--rlabel', default=False, type=bool, help='rlabel')
    # parser.add_argument('--arch', default=None, required=True, type=str, help='Vision model.')
    # parser.add_argument('--data', default=None, required=True, type=str, help='Vision dataset.')
    # parser.add_argument('--epochs', default=None, required=True, type=int, help='Vision epoch.')
    parser.add_argument('--resume', default=-1, type=int, help='rlabel')
    parser.add_argument('--num_images', default=1, type=int, help='Number of images to do simultaneously.')
    # parser.add_argument('--moment_matching', default=False, action='store_true', help='Moment matching.')
    # parser.add_argument('--affine_transform', default=False, action='store_true', help='Affine transform.')
    # parser.add_argument('--translation_transform', default=False, action='store_true', help='Translation transform.')
    # parser.add_argument('--translation_clip', default=(2.0, 2.0), type=float, help='Translation clip.', nargs=2,
    #                     metavar=('x', 'y'))
    # parser.add_argument('--shear_clip', default=(0.1, 0.1), type=float, help='Shear clip. Clips the shear values of the affine transformation matrix.', nargs=2, metavar=('x', 'y'))
    # parser.add_argument('--scale_clip', default=(0.1, 0.1), type=float, help='Scale clip. Clips the scale values (diagonal) of the affine transformation matrix. The default values clip these to the range 0.9-1.1', nargs=2, metavar=('x', 'y'))
    # parser.add_argument('--shift_left', default=False, action='store_true', help='Shift left.')
    # parser.add_argument('--shift_right', default=False, action='store_true', help='Shift right.')
    parser.add_argument('--reaugment', default='none', type=str, help='Reaugmentation mode.')


def get_args():
    parser = argparse.ArgumentParser(description='yo')

    parse_search_args(parser)
    parse_single_policy(parser)
    parse_attack(parser)

    args = parser.parse_args()

    return args
