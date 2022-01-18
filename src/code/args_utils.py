import argparse
from xmlrpc.client import Boolean

def get_args():
    parser = argparse.ArgumentParser(description='yo')

        # POLICY SEARCH SETTINGS
    parser.add_argument('--check_if_exists', type=bool, default=True)
    parser.add_argument('--num_random_policies_to_test', type=int, default=50)

    parser.add_argument('--augmentations_per_policy', type=int, default=3)
    parser.add_argument('--num_possible_augmentations', type=int, default=50)


    # SINGLE POLICY SETTINGS
    parser.add_argument('--aug_list', type=str, default='45-36-28')

    parser.add_argument('--dataset_name', type=str, default='cifar100')
    parser.add_argument('--data_path', type=str, default='./data')

    parser.add_argument('--transform_mode', type=str, default='aug')
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--augment_validation', type=Boolean, default=True)

    parser.add_argument('--architecture', type=str, default='ResNet20-4')
    parser.add_argument('--model_trained_partially_path', type=str, default='./models/ResNet20-4.pth')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epochs', type=int, default=100)

    parser.add_argument('--recon_tests_per_policy', type=int, default=100)
    parser.add_argument('--num_images_per_recon_test', type=int, default=1)

    parser.add_argument('--results_dir', type=str, default='./results')

    args = parser.parse_args()

    return args
