

from torch.utils.data import Subset
from torch.utils.data import DataLoader, Subset

def create_subset_data(trainloader_cub200, testloader_cub200, batch_size=32):
    # Create subsets
    n_train_samples = 300  # or whatever number you want
    n_test_samples = 50  # or whatever number you want

    train_subset = Subset(trainloader_cub200.dataset, range(0, n_train_samples))
    test_subset = Subset(testloader_cub200.dataset, range(0, n_test_samples))

    # Create new dataloaders
    trainloader_small = DataLoader(train_subset, batch_size=batch_size, num_workers=10)
    testloader_small = DataLoader(test_subset, batch_size=batch_size, num_workers=10)

    return trainloader_small, testloader_small


