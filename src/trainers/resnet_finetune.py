import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from models.resnet50 import ResNet50

class FineTune(nn.Module):
    def __init__(self, model, trainloader, testloader, criterion, optimizer, device):
        self.model = model
        self.trainloader = trainloader
        self.valloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        
        if self.device == None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    def train_one_epoch(self):
        self.model.train()
        running_loss = 0.0
        for images, labels in self.trainloader:
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero the parameter gradients
            self.optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            # Print statistics
            running_loss += loss.item()
        print(f'Training loss: {running_loss / len(self.trainloader)}')

    def validate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        with torch.inference_mode():
            for images, labels in self.testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                
                # _, predicted = torch.max(outputs.data, 1)
                predicted = torch.argmax(outputs.data, dim=1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
        print(f'Finetuning validation loss: {running_loss / len(self.testloader)}')
        print(f'Finetuning validation accuracy: {100 * correct / total}%')

    def fine_tune(self, num_epochs, save_path):
        best_acc = 0.0
        for epoch in range(num_epochs):
            print(f'Epoch {epoch + 1}/{num_epochs}')
            self.train_one_epoch()
            self.validate()
            val_acc = self.validate()
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
        print('Finished Fine Tuning')
