
# write a custom dataloader class for the cub200 dataset
from datasets.cub200 import Cub2011
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import v2 
# import Cutmix
# import torchvision.transforms.functional as TF


# preserve dimensions of images
# def pad(img, size_max=500):
#     """
#     Pads images to the specified size (height x width). 
#     """
#     pad_height = max(0, size_max - img.height)
#     pad_width = max(0, size_max - img.width)
    
#     pad_top = pad_height // 2
#     pad_bottom = pad_height - pad_top
#     pad_left = pad_width // 2
#     pad_right = pad_width - pad_left
    
#     return TF.pad(
#         img,
#         (pad_left, pad_top, pad_right, pad_bottom),
#         fill=tuple(map(lambda x: int(round(x * 256)), (0.485, 0.456, 0.406))))


class DataLoaderCUB200:
    def __init__(self, data_root, batch_size=32, num_workers=10):
        """
        Initializes the DataLoaderCUB200 class with specified batch size, number of workers, and data root.
        """
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        # old augmentation strategy that yielded 61.59% accuracy after 9 epochs (early stopping) on 2048 feature size
        # self.train_transform = transforms.Compose([
        #     transforms.Resize((224, 224)),  # Resize images to the size expected by ResNet50
        #     transforms.RandomHorizontalFlip(p=0.5), # p = probability of flip, 0.5 = 50% chance
        #     transforms.ToTensor(),  # Convert PIL image to Tensor
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalize to ImageNet stats
        # ])

        # # augmentation strategy from the Deit paper

        # self.train_transform = v2.Compose([
        #     v2.Resize((224, 224)),
        #     v2.TrivialAugmentWide(num_magnitude_bins=31), # how intense 
        #     v2.ToTensor() # use ToTensor() last to get everything between 0 & 1
        # ])

        self.train_transform = v2.Compose([
            # v2.Lambda(pad),
            v2.Resize(256),
            v2.RandomResizedCrop(224),
            v2.RandomHorizontalFlip(),
            v2.TrivialAugmentWide(),            
            # v2.CutMix(cutmix_alpha=1.0, num_classes=200),
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        # # new strategy
        # self.train_transform = transforms.Compose([
        #     transforms.RandomResizedCrop(224),  # Random crop and resize
        #     transforms.RandomHorizontalFlip(),
        #     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        #     transforms.RandomRotation(20),  # Random rotation by up to 20 degrees
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # ])


        self.test_transform = transforms.Compose([
            # v2.Lambda(pad),
            v2.Resize((224, 224)),  # Resize images to the size expected by ResNet50
            v2.CenterCrop(256),  # Center crop the image
            v2.ToTensor(),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.train_dataset = Cub2011(self.data_root, train=True, download=False, transform=self.train_transform)
        self.test_dataset = Cub2011(self.data_root, train=False, download=False, transform=self.test_transform)

    # In your DataLoaderCUB200 class
    def get_number_of_classes(self):
        # Assuming that the number of classes is the same in both train and test datasets
        return self.train_dataset.get_number_of_classes()


    def get_dataloaders(self):
        """
        Creates and returns data loaders for the CUB-200-2011 dataset (both training and testing sets).
        
        Returns:
        tuple: A tuple containing the training and testing data loaders.
        """
        trainloader = DataLoader(dataset=self.train_dataset, 
                                batch_size=self.batch_size, 
                                num_workers=self.num_workers, 
                                shuffle=True)

        testloader = DataLoader(dataset=self.test_dataset, 
                                batch_size=self.batch_size, 
                                num_workers=self.num_workers, 
                                shuffle=False)

        return trainloader, testloader
