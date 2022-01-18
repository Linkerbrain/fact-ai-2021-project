import inversefed

def create_model(dataset_name, architecture):
    """
    Creates an `architecture` type model that fits onto `dataset_name`
    """
    if dataset_name == 'cifar100':
        model, _ = inversefed.construct_model(architecture, num_classes=100, num_channels=3)
    elif dataset_name == 'FashionMnist':
        model, _ = inversefed.construct_model(architecture, num_classes=10, num_channels=1)
    elif dataset_name == 'Mnist':
        model, _ = inversefed.construct_model(architecture, num_classes=10, num_channels=1)
    else:
        raise NotImplementedError()

    return model
