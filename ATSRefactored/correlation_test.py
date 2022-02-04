import copy, random
from tqdm import tqdm
import os

from args_utils import get_args
from correlation_evaluator import CorrelationEvaluator

class CorrelationTest:
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

    def perform_test(self, policies):
        """
        """
        for policy in tqdm(policies, desc="POLICY SEARCHER PROGRESS:"):
            # check if exists
            if self.check_if_exists:
                directory = os.path.join(self.results_dir, 'correlation/data_{}_arch_{}/'.format(self.dataset_name, self.architecture))
                file = os.path.join(directory, f'{policy}_privacy_score.npy')
                if os.path.isfile(file):
                    print(f"> {policy} results already exist! Skipping..")
                    continue

            # make evaluator
            self.args.aug_list = policy
            evaluator = CorrelationEvaluator(self.args)

            # evaluate
            evaluator.evaluate_correlation()

def main():
    # init
    args = get_args()
    tester = CorrelationTest(args)

    # make list of policies to test
    random_policies = tester.make_policy_list()
    guarenteed_policies = []

    # evaluate all policies
    tester.perform_test(guarenteed_policies + random_policies)


if __name__ == "__main__":
    main()
