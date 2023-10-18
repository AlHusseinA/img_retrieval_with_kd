# import packages and modules
from torchvision import datasets, transforms
from torch.utils.data import DataLoader




class DataLoaderCIFAR10:
    def __init__(self, batch_size=32, data_root='./data'):
        """
            Initializes the data loader with a specified batch size and data root directory.
            
            Args:
            batch_size (int): The batch size for the data loader.
            data_root (str): The directory where the data should be downloaded and stored.

        """        

        self.batch_size = batch_size
        self.data_root = data_root
        
        self.transform = transforms.Compose([transforms.Resize(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                  std=[0.229, 0.224, 0.225])])

    def get_dataloaders(self):
        """
            Creates and returns data loaders for the CIFAR-10 dataset (both training and testing sets).
            
            Returns:
            tuple: A tuple containing the training and testing data loaders.
        """

        trainset = datasets.CIFAR10(root=self.data_root, train=True, download=True, transform=self.transform)
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testset = datasets.CIFAR10(root=self.data_root, train=False, download=True, transform=self.transform)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        num_classes = len(trainset.classes)

        return trainloader, testloader, num_classes