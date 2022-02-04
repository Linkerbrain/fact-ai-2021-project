import torch

def create_config(attack_type, max_iters=4800):
    if attack_type == 'inversed':
        config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=max_iters,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif attack_type == 'inversed-zero':
        config = dict(signed=True,
            boxed=True,
            cost_fn='sim',
            indices='def',
            weights='equal',
            lr=0.1,
            optim='adam',
            restarts=1,
            max_iterations=max_iters,
            total_variation=1e-4,
            init='zeros',
            filter='none',
            lr_decay=True,
            scoring_choice='loss')
    elif attack_type == 'inversed-sim-out':
        config = dict(signed=True,
            boxed=True,
            cost_fn='out_sim',
            indices='def',
            weights='equal',
            lr=0.1,
            optim='adam',
            restarts=1,
            max_iterations=max_iters,
            total_variation=1e-4,
            init='zeros',
            filter='none',
            lr_decay=True,
            scoring_choice='loss')
    elif attack_type == 'inversed-sgd-sim':
        config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='sgd',
                restarts=1,
                max_iterations=max_iters,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif attack_type == 'inversed-LBFGS-sim':
        config = dict(signed=True,
                boxed=True,
                cost_fn='sim',
                indices='def',
                weights='equal',
                lr=1e-4,
                optim='LBFGS',
                restarts=16,
                max_iterations=300,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=False,
                scoring_choice='loss')
    elif attack_type == 'inversed-adam-L1':
        config = dict(signed=True,
                boxed=True,
                cost_fn='l1',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=max_iters,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif attack_type == 'inversed-adam-L2':
        config = dict(signed=True,
                boxed=True,
                cost_fn='l2',
                indices='def',
                weights='equal',
                lr=0.1,
                optim='adam',
                restarts=1,
                max_iterations=max_iters,
                total_variation=1e-4,
                init='randn',
                filter='none',
                lr_decay=True,
                scoring_choice='loss')
    elif attack_type == 'zhu':
        config = dict(signed=False,
                        boxed=False,
                        cost_fn='l2',
                        indices='def',
                        weights='equal',
                        lr=1e-4,
                        optim='LBFGS',
                        restarts=2,
                        max_iterations=50, # ??
                        total_variation=1e-3,
                        init='randn',
                        filter='none',
                        lr_decay=False,
                        scoring_choice='loss')
        seed=1
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        import random
        random.seed(seed)
    else:
        raise NotImplementedError
    return config