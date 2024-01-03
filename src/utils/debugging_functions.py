

from torch.utils.data import Subset
from torch.utils.data import DataLoader, Subset

def create_subset_data(trainloader_cub200, testloader_cub200, batch_size=32):
    # Create subsets
    n_train_samples = 300  # or whatever number you want
    n_test_samples = 200  # or whatever number you want

    train_subset = Subset(trainloader_cub200.dataset, range(0, n_train_samples))
    test_subset = Subset(testloader_cub200.dataset, range(0, n_test_samples))

    # Create new dataloaders
    trainloader_small = DataLoader(train_subset, batch_size=batch_size, num_workers=10)
    testloader_small = DataLoader(test_subset, batch_size=batch_size, num_workers=10)

    return trainloader_small, testloader_small, batch_size


def create_subset_data2(trainloader_cub200, testloader_cub200, subset_ratio, batch_size=32):
    # Validate and adjust the subset_ratio
    if subset_ratio <= 0:
        subset_ratio = 0.1  # Set to 10% if zero or negative
    elif subset_ratio > 1:
        subset_ratio = 1  # Cap at 100% if more than 100%

    # Determine the number of samples to use
    n_train_samples = int(len(trainloader_cub200.dataset) * subset_ratio)
    n_test_samples = int(len(testloader_cub200.dataset) * subset_ratio)

    # Ensure at least 1 sample is used if the dataset is not empty
    n_train_samples = max(1, n_train_samples)
    n_test_samples = max(1, n_test_samples)

    # Create subsets
    train_subset = Subset(trainloader_cub200.dataset, range(0, n_train_samples))
    test_subset = Subset(testloader_cub200.dataset, range(0, n_test_samples))

    # Create new dataloaders
    trainloader_small = DataLoader(train_subset, batch_size=batch_size, num_workers=10)
    testloader_small = DataLoader(test_subset, batch_size=batch_size, num_workers=10)

    return trainloader_small, testloader_small, batch_size
