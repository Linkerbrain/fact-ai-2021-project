import copy, random
from tqdm import tqdm
import os

from args_utils import get_args
from policy_evaluator import PolicyEvaluator

class PolicySearcher:
    def __init__(self, args):
        self.args = args

        self.results_dir = args.results_dir
        self.dataset_name = args.dataset_name
        self.architecture = args.architecture

        self.num_random_policies_to_test = args.num_random_policies_to_test
        self.augmentations_per_policy = args.augmentations_per_policy

        self.num_possible_augmentations = args.num_possible_augmentations

        self.check_if_exists = args.check_if_exists

    def make_policy_list(self):
        """
        Creates `num_random_policies_to_test` of random policies
        consisting of `augmentations_per_policy` augmentations
        """
        policies = []

        for _ in range(self.num_random_policies_to_test):
            policy = []

            # select random policies
            for _ in range(self.augmentations_per_policy):
                policy.append(random.randint(-1, self.num_possible_augmentations-1))

            # -1 indicates no augmentation, so we remove its occurences
            policy = list(filter((-1).__ne__, policy))

            policies.append("-".join([str(a) for a in policy]))

        return policies

    def evaluate_policies(self, policies):
        """
        Evaluates all policies, saves result to disk
        """
        for policy in tqdm(policies, desc="POLICY SEARCHER PROGRESS:"):
            # check if exists
            if self.check_if_exists:
                recon_dir = os.path.join(self.results_dir, 'reconstruction/data_{}_arch_{}/'.format(self.dataset_name, self.architecture))
                recon_file = os.path.join(recon_dir, f'{policy}.npy')
                acc_dir = os.path.join(self.results_dir, 'accuracy/data_{}_arch_{}/'.format(self.dataset_name, self.architecture))
                acc_file = os.path.join(acc_dir, f'{policy}.npy')
                if os.path.isfile(recon_file) and os.path.isfile(acc_file):
                    print(f"> {policy} results already exist! Skipping..")
                    continue

            # make evaluator
            self.args.aug_list = policy
            evaluator = PolicyEvaluator(self.args)

            # evaluate
            evaluator.evaluate_policy()

def main():
    # init
    args = get_args()
    policy_searcher = PolicySearcher(args)

    # make list of policies to test
    random_policies = policy_searcher.make_policy_list()

    guarenteed_policies = [
        "3-1-7",
        "43-18-18",
        "21-13-3",
        "7-4-15"
    ]

    # evaluate all policies
    policy_searcher.evaluate_policies(guarenteed_policies + random_policies)


if __name__ == "__main__":
    main()
