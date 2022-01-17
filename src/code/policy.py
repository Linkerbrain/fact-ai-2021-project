import numpy as np
from available_policies import available_policies

class Policy:
    def __init__(self, policy_list):
        # hybrid policy
        if isinstance(policy_list[0], list):
            self.policy_list = policy_list
        
        # regular policy
        elif isinstance(policy_list[0], int):
            self.policy_list = [policy_list]

        else:
            raise NotImplementedError

    def __call__(self, img):
        # select a random policy (if multiple policies are present, used for hybrid)
        idx = np.random.randint(0, len(self.policy_list))
        select_policy = self.policy_list[idx]

        # apply policy
        for policy_id in select_policy:
            img = available_policies[policy_id](img)

        return img

    def __str__(self):
        if len(self.policy_list) == 1:
            return "Policy with Augmentations: " + str(self.policy_list[0])
        else:
            return "Hybrid policy: " + str(self.policy_list)