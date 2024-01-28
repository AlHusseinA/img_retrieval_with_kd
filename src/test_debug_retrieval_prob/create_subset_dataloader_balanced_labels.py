from torch.utils.data import DataLoader, Subset
from random import sample
from datasets.cub200 import Cub2011
from torchvision import transforms
from torchvision.transforms import v2 

def filter_dataloader_cub200(data_root, batch_size, num_workers, n_labels=10, images_per_label=10):
    """
    Filters the CUB-200-2011 dataset to include only n_labels for both training and testing datasets, 
    each with a fixed number of images.

    :param data_root: The root directory of the dataset.
    :param batch_size: Batch size for the DataLoader.
    :param num_workers: Number of workers for the DataLoader.
    :param n_labels: Number of labels to include.
    :param images_per_label: Number of images per label.
    :return: Two DataLoaders, one for the filtered training dataset and another for the filtered testing dataset.
    """



    train_transform = v2.Compose([
        v2.Resize(256),
        v2.RandomResizedCrop(224),
        v2.RandomHorizontalFlip(),
        v2.TrivialAugmentWide(),            
        v2.ToTensor(), 
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        v2.Resize((224, 224)),  # Resize images to the size expected by ResNet50
        v2.CenterCrop(256),  # Center crop the image
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])    
        
    # Initialize the Cub2011 datasets for training and testing
    train_dataset = Cub2011(root=data_root, train=True, download=False, transform=train_transform)
    test_dataset = Cub2011(root=data_root, train=False, download=False, transform=test_transform)

    # Function to create a subset of the dataset
    def create_subset(dataset):
        if not hasattr(dataset, 'targets'):
            raise ValueError("Dataset must have 'targets' attribute")
        
        # labels from 0 to n_labels-1
        label_indices = {label: [] for label in range(n_labels)}

        for idx, label in enumerate(dataset.targets):
            if label < n_labels:
                label_indices[label].append(idx)

        selected_indices = [sample(label_indices[label], min(len(label_indices[label]), images_per_label))
                            for label in label_indices]
        # print(f"selected_indices before summing: {selected_indices=}")
        selected_indices = sum(selected_indices, [])    
        # print(f"selected_indices sum: {selected_indices}")

        return Subset(dataset, selected_indices)

    # Create subsets
    subset_train_dataset = create_subset(train_dataset)
    subset_test_dataset = create_subset(test_dataset)

    # Create DataLoaders for the subsets
    filtered_loader_train = DataLoader(subset_train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    filtered_loader_test = DataLoader(subset_test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return filtered_loader_train, filtered_loader_test

# Example usage
# filtered_loader_train, filtered_loader_test = filter_dataloader_cub200(data_root, batch_size=10, num_workers=10, n_labels=10)
