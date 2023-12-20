import torch
import torch.nn as nn


# write a concise modular trainer class that takes in a model, feature size, optimizer, loss, trainloader, testloader, device, and number of epochs
class BaseTrainer:
    def __init__(self, model, optimizer, criterion, trainloader, testloader, device, epochs=10):
        """
            Initializes the trainer with a model, optimizer, loss, trainloader, testloader, device, and number of epochs.

            Args:
            model (torch.nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): The optimizer to be used for training.
            criterion (torch.nn.Module): The loss function to be used for training.
            trainloader (torch.utils.data.DataLoader): The dataloader for training.
            testloader (torch.utils.data.DataLoader): The dataloader for testing.
            device (str): The device to be used for training.
            epochs (int): The number of epochs to train the model for.
        """
        self.model = model
        # self.feature_size = feature_size
        self.optimizer = optimizer
        self.criterion = criterion
        self.trainloader = trainloader
        self.testloader = testloader
        self.device = device
        self.epochs = epochs

    def train(self):
        """
            Trains the model for the specified number of epochs.
        """
        self.model.train()
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1} of {self.epochs}")
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()

                self.optimizer.step()

                if i % 100 == 0:
                    print(f"Loss at iteration {i} of epoch {epoch+1} is {loss.item()}")

    def test(self):
        """
            Tests the model on the test set.
        """
        self.model.eval()
        correct = 0
        total = 0
        with torch.inference_mode():
            for data in self.testloader:
                images, labels = data
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total

        # write a test evaluate method that evaluates 