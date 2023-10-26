import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

class ResnetTrainer_test:
    def __init__(self, resnet50, criterion, optimizer, device, use_early_stopping, scheduler=None):
        
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.resnet50 = resnet50.to(device)  # Move model to the specified device
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.scheduler = scheduler
        self.min_epochs_for_early_stopping = 8
        self.actual_epochs_run = 0
        self.use_early_stopping= use_early_stopping

        if self.use_early_stopping:
            self.max_val_accuracy = 0.0  # Initialize maximum validation accuracy to 0
            self.patience = 20 #3  # Number of epochs to wait before stopping

            self.counter = 0  # Counter to keep track of epochs since an improvement in validation accuracy


    def check_early_stopping(self, avg_val_accuracy, counter, patience):
        if self.actual_epochs_run < self.min_epochs_for_early_stopping:
            return False  # Continue training
        
        if avg_val_accuracy > self.max_val_accuracy:
            self.max_val_accuracy = avg_val_accuracy
            self.counter = 0  # Reset counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch: {self.actual_epochs_run}.")

                # insert logic for config file

                return True  # Stop training
        return False  # Continue training
    

    # def train(self, train_loader, val_loader, num_epochs):
    #     for epoch in range(num_epochs):
    #         train_loss = 0.0
    #         train_acc = 0.0
    #         val_loss = 0.0
    #         val_acc = 0.0

    #         # Training phase
    #         self.resnet50.train()
    #         for i, (inputs, labels) in enumerate(train_loader):
    #             inputs, labels = inputs.to(self.device), labels.to(self.device)  # Move data to the specified device

    #             self.optimizer.zero_grad()
    #             outputs = self.resnet50(inputs)
    #             loss = self.criterion(outputs, labels)
    #             loss.backward()
    #             self.optimizer.step()

    #             train_loss += loss.item()
    #             _, predicted = torch.max(outputs.data, 1)
    #             train_acc += (predicted == labels).sum().item()
    #         if self.scheduler is not None:
    #             self.scheduler.step()
    #         train_loss /= len(train_loader)
    #         train_acc /= len(train_loader)
    #         self.train_losses.append(train_loss)
    #         self.train_accs.append(train_acc)

    #         # Validation phase
    #         self.resnet50.eval()
    #         with torch.no_grad():
    #             for inputs, labels in val_loader:
    #                 inputs, labels = inputs.to(self.device), labels.to(self.device)  # Move data to the specified device

    #                 outputs = self.resnet50(inputs)
    #                 loss = self.criterion(outputs, labels)

    #                 val_loss += loss.item()
    #                 _, predicted = torch.max(outputs.data, 1)
    #                 val_acc += (predicted == labels).sum().item()

    #             val_loss /= len(val_loader)
    #             val_acc /= len(val_loader)
    #             self.val_losses.append(val_loss)
    #             self.val_accs.append(val_acc)

    #         self.actual_epochs_run += 1
    #         if self.use_early_stopping:
    #             stop_training = self.check_early_stopping(val_acc, self.counter, self.patience)
    #             if stop_training:
    #                 break  # Stop training

    #         print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
    #     # Return the fine-tuned model for further use
    #     return self.resnet50
    
    def train(self, train_loader, val_loader, num_epochs):
        # Initialize counters for the total number of correct predictions and the total number of samples
        total_correct_train = 0
        total_samples_train = 0
        total_correct_val = 0
        total_samples_val = 0

        for epoch in range(num_epochs):
            train_loss = 0.0
            val_loss = 0.0

            # Training phase
            self.resnet50.train()
            for i, (inputs, labels) in enumerate(train_loader):
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.resnet50(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_correct_train += (predicted == labels).sum().item()
                total_samples_train += labels.size(0)

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss /= len(train_loader)
            train_acc = (total_correct_train / total_samples_train) * 100

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validation phase
            self.resnet50.eval()
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    outputs = self.resnet50(inputs)
                    loss = self.criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total_correct_val += (predicted == labels).sum().item()
                    total_samples_val += labels.size(0)

                val_loss /= len(val_loader)
                val_acc = (total_correct_val / total_samples_val) * 100

                self.val_losses.append(val_loss)
                self.val_accs.append(val_acc)

            self.actual_epochs_run += 1

            if self.use_early_stopping:
                stop_training = self.check_early_stopping(val_acc, self.counter, self.patience)
                if stop_training:
                    break

            print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

        return self.resnet50

    

    def plot_loss_vs_epoch(self):
        fig, ax = plt.subplots()
        epochs = range(1, len(self.train_losses) + 1)
        
        # Plot the data
        ax.plot(epochs, self.train_losses, label='Train')
        ax.plot(epochs, self.val_losses, label='Validation')
        
        # Find the epoch and value for minimum training loss
        min_train_loss_epoch = epochs[self.train_losses.index(min(self.train_losses))]
        min_train_loss_value = min(self.train_losses)
        
        # Find the epoch and value for minimum validation loss
        min_val_loss_epoch = epochs[self.val_losses.index(min(self.val_losses))]
        min_val_loss_value = min(self.val_losses)
        
        # Annotate the points for minimum training and validation loss
        ax.annotate(f'{min_train_loss_value:.4f}', xy=(min_train_loss_epoch, min_train_loss_value),
                    xytext=(min_train_loss_epoch, min_train_loss_value),
                    arrowprops=dict(facecolor='red', arrowstyle='->'),
                    fontsize=12)
        
        ax.annotate(f'{min_val_loss_value:.4f}', xy=(min_val_loss_epoch, min_val_loss_value),
                    xytext=(min_val_loss_epoch, min_val_loss_value),
                    arrowprops=dict(facecolor='red', arrowstyle='->'),
                    fontsize=12)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss vs Epoch')
        ax.legend()
        
        return fig

    def plot_acc_vs_epoch(self):
        fig, ax = plt.subplots()
        epochs = range(1, len(self.train_accs) + 1)
        
        # Plot the data
        ax.plot(epochs, self.train_accs, label='Train')
        ax.plot(epochs, self.val_accs, label='Validation')
        
        # Find the epoch and value for maximum training accuracy
        max_train_acc_epoch = epochs[self.train_accs.index(max(self.train_accs))]
        max_train_acc_value = max(self.train_accs)
        
        # Find the epoch and value for maximum validation accuracy
        max_val_acc_epoch = epochs[self.val_accs.index(max(self.val_accs))]
        max_val_acc_value = max(self.val_accs)
        
        # Annotate the points for maximum training and validation accuracy
        ax.annotate(f'{max_train_acc_value:.2f}%', xy=(max_train_acc_epoch, max_train_acc_value),
                    xytext=(max_train_acc_epoch, max_train_acc_value),
                    arrowprops=dict(facecolor='red', arrowstyle='->'),
                    fontsize=12)
        
        ax.annotate(f'{max_val_acc_value:.2f}%', xy=(max_val_acc_epoch, max_val_acc_value),
                    xytext=(max_val_acc_epoch, max_val_acc_value),
                    arrowprops=dict(facecolor='red', arrowstyle='->'),
                    fontsize=12)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy vs Epoch')
        ax.legend()
        
        return fig