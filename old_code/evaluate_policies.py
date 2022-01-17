import copy, random

from policy_evaluator import PolicyEvaluator
from tqdm import tqdm

"""
SETTINGS
"""
NUM_POLICIES = 50
NUM_AUGMENTATIONS = 3
NUM_POSSIBLE_AUGMENTATIONS = 50

ARCH = "ResNet20-4"
DATA = "cifar100"


def create_policy_queue(num_policies, num_augmentations, num_possible_augmentations=50):
    """
    Samples policies consisting of up to `num_augmentations` augmentation ids
    """
    policies = []

    for _ in range(num_policies):
        policy = []

        # select random policies
        for _ in range(num_augmentations):
            policy.append(random.randint(-1, num_possible_augmentations))

        # -1 indicates no augmentation, so we remove its occurences
        policy = list(filter((-1).__ne__, policy))

        policies.append("-".join([str(a) for a in policy]))

    return policies


def evaluate_policies(policies):
    """
    Evaluates a policy quickly based on 2 things
    """

    for policy in tqdm(policies, desc="POLICIES DONE:"):
        evaluator = PolicyEvaluator(
            mode="aug",
            aug_list=policy,
            rlabel=False,
            arch=ARCH,
            data=DATA,
            epochs=100,
            iters_per_policy=100,
        )
        evaluator.evaluate_policy()


def main():
    print("Generating policies..")
    policies = create_policy_queue(
        NUM_POLICIES, NUM_AUGMENTATIONS, NUM_POSSIBLE_AUGMENTATIONS
    )
    print(f"Generated {len(policies)} policies!")

    print("Evaluating policies..")
    evaluate_policies(policies)
    print("Done!")


if __name__ == "__main__":
    main()
