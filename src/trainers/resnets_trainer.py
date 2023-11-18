import torch
import torchmetrics
import yaml

import matplotlib.pyplot as plt
import torch.nn as nn
from exp_logging.metricsLogger import MetricsLogger
import torchmetrics
from torchmetrics.classification import Accuracy
# from utils.helper_functions import plot_multiple_metrics

class ResnetTrainer():
    def __init__(self, model, optimizer, criterion,lr,
                 scheduler, trainloader, testloader, 
                 feature_size, use_early_stopping,
                 device, num_classes, log_save_path, metrics_logger=None,
                 epochs=10, dataset_name=None):
             

        self.model = model
        
        self.metrics_logger = metrics_logger
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.dataset_name = dataset_name if dataset_name else "Unknown"
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.lr = lr
        self.trainloader = trainloader
        self.testloader = testloader
        self.epochs = epochs
        self.log_save_path = log_save_path
        self.use_early_stopping = use_early_stopping
        self.metric = Accuracy(task="multiclass", num_classes=self.num_classes)  # Adjust num_classes as per your dataset
        self.metric_eval = Accuracy(task="multiclass", num_classes=self.num_classes) 
        self.device = device
        self.actual_epochs_run = 0
        self.min_epochs_for_early_stopping = 15
        if device:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        self.model.to(self.device)
        self.metric.to(self.device)
        self.metric_eval.to(self.device)
       
       
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
            if counter >= patience:
                print(f"\nEarly stopping triggered at epoch: {self.actual_epochs_run}.")

                # insert logic for config file

                return True  # Stop training
        return False  # Continue training

    def train_model(self):
 
        self.accumulated_val_losses = []
        self.accumulated_val_accuracies = []
        self.training_accuracy = []  # Initialize a list to store the training accuracy per epoch
        self.training_loss_list = []  # Initialize a list to store the training loss per epoch
        # Inside train_model
        batch_accuracies = []  # List to store batch-wise accuracies

        for epoch in range(self.epochs):
            self.model.train()
            training_loss = 0.0
            # is_last_batch = False 
            for batch_idx, data in enumerate(self.trainloader):
            # for data in self.trainloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)                
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()               
                training_loss += loss.item()
            
                # Calculating accuracy using torchmetrics inside the loop for each batch
                acc = self.metric(outputs.softmax(dim=-1), labels)
                # more compactly
                # self.metric(outputs.softmax(dim=-1), labels)
                batch_accuracies.append(acc.item() * 100)  # Storing batch-wise accuracy

            if self.scheduler is not None:
                self.scheduler.step()
                
            average_validation_loss, average_validation_accuracy = self.evaluate_model()
            # they are get appeneded so they can be plotted later
            self.accumulated_val_losses.append(average_validation_loss)
            self.accumulated_val_accuracies.append(average_validation_accuracy)
            self.training_loss_list.append(training_loss / len(self.trainloader))
            epoch_accuracy = self.metric.compute().item() * 100  # Compute the accumulated accuracy
            # the above can be re-written as
            # epoch_accuracy = acc.compute().item() * 100
            self.training_accuracy.append(epoch_accuracy)
            self.actual_epochs_run += 1


            if self.use_early_stopping:
                stop_training = self.check_early_stopping(average_validation_accuracy, self.counter, self.patience)
                if stop_training:
                    break  # Stop training

            # self.training_loss.append(training_loss / len(self.trainloader.dataset))
            # print(f'Epoch {epoch+1}, Training Loss: {training_loss / len(self.trainloader):.4f}, Training Accuracy: {epoch_accuracy:.4f}%')
            print(f'Epoch {epoch + 1}, Training Loss: {training_loss / len(self.trainloader):.4f}, Training Accuracy: {epoch_accuracy:.4f}, Validation Loss: {average_validation_loss:.4f}, Validation Accuracy: {average_validation_accuracy:.4f}')


            if self.metrics_logger is not None:
                try:
                    self.metrics_logger.log_metrics(self.feature_size, self.actual_epochs_run, training_loss, self.training_accuracy, average_validation_loss, average_validation_accuracy, self.dataset_name, self.log_save_path, self.lr)

                    # self.metrics_logger.log_metrics(self.feature_size, epoch, training_loss, self.training_accuracy, average_validation_loss, average_validation_accuracy, batch_accuracies, self.dataset_name)
                except TypeError as te:
                    print(f"TypeError occurred while logging metrics: {te}")
                    print(f"Check the data types of variables being logged. Current types: {type(self.feature_size)}, {type(epoch)}, {type(training_loss)}, etc.")
                except ValueError as ve:
                    print(f"ValueError occurred while logging metrics: {ve}")
                    print(f"Check the value validity of variables being logged. Current values: {self.feature_size}, {epoch}, {training_loss}, etc.")
                except Exception as e:
                    print(f"An unexpected error occurred while logging metrics: {e}")
                    print(f"State of variables at the time of error: feature_size={self.feature_size}, epoch={epoch}, training_loss={training_loss}, etc.")
                    import traceback
                    print("Stack Trace:")
                    print(traceback.format_exc())

            
            self.metric.reset()  # Reset the metric for the next epoch
        
        return self.model, self.training_loss_list, self.training_accuracy, self.accumulated_val_losses, self.accumulated_val_accuracies        
    
    def evaluate_model(self):        
        # set model to evaluation mode
        self.model.eval()
        total_batches = len(self.testloader)
        tenth_of_batches = total_batches // 10  # Calculate 10% of total_batches

        running_loss = 0.0
        # self.metric_eval
        
        with torch.inference_mode():
            # for images, labels in self.testloader:
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                # add up loss for each batch, running_loss will be just a float at the end of the for loop
                # accumulate loss for each batch
                running_loss += loss.item()

                self.metric_eval(outputs.softmax(dim=-1), labels)
                # batch_accuracies.append(acc_val.item() * 100)  # Storing val batch-wise accuracy
                # Clone metric object and compute for this batch
                metric_clone = self.metric_eval.clone()
                accuracy_this_batch = metric_clone.compute().item() * 100  # Note: this assumes the metric is between 0 and 1

                
            # Compute the accumulated metrics
            epoch_accuracy_eval = self.metric_eval.compute().item() * 100
            avg_loss_over_batch = running_loss / total_batches
            
            # print(f'Val Epoch, Validation Loss: {avg_loss_over_batch:.4f}, Validation Accuracy: {epoch_accuracy_eval:.2f}%')

        avg_loss_over_batch = running_loss / total_batches

        # accuracy_over_epoch should be a 
        return avg_loss_over_batch, epoch_accuracy_eval  # Return the list of batch losses
    

    # def plot_metrics(self, train_losses, test_losses, train_accuracies, test_accuracies):
    #     # x-axis is actual epochs run not pre-defined epochs since we're using early stopping
    #     epochs_range = range(self.actual_epochs_run)
    #     plot_id = f"{self.dataset_name}_ResNet50_FeatureSize_{self.feature_size}"
    
    #     # Function to plot generic metrics
    #     def plot_generic_metrics(epochs_range, train_metric, test_metric, metric_name, ylabel):
    #         if len(epochs_range) != len(train_metric) or len(epochs_range) != len(test_metric):
    #             raise ValueError(f"Mismatch in lengths: epochs_range ({len(epochs_range)}), train_metric ({len(train_metric)}), test_metric ({len(test_metric)})")

    #         plt.figure(figsize=(14, 10))  
    #         plt.title(f"Feature size {self.feature_size}")
    #         plt.plot(epochs_range, train_metric, label=f'Training {metric_name}')
    #         plt.plot(epochs_range, test_metric, label=f'Testing {metric_name}')
    #         plt.xlabel('Epochs')
    #         plt.ylabel(ylabel)
    #         plt.title(f'Training and Testing {metric_name} over Epochs - {plot_id}')
    #         plt.legend()
    #         plt.savefig(f"{self.log_save_path}_{metric_name}_{plot_id}.png")
    #         # plt.show()


    #     # Plot training and validation loss over epochs
    #     plot_generic_metrics(epochs_range, train_losses, test_losses, 'Loss', 'Loss')

    #     # Plot training and validation accuracy over epochs
    #     plot_generic_metrics(epochs_range, train_accuracies, test_accuracies, 'Accuracy', 'Accuracy (%)')


    def plot_metrics(self, train_losses, test_losses, train_accuracies, test_accuracies):
        epochs_range = range(self.actual_epochs_run)
        plot_id = f"{self.dataset_name}_ResNet50_FeatureSize_{self.feature_size}"

        def plot_generic_metrics(epochs_range, train_metric, test_metric, metric_name, ylabel):
            if len(epochs_range) != len(train_metric) or len(epochs_range) != len(test_metric):
                raise ValueError(f"Mismatch in lengths: epochs_range ({len(epochs_range)}), train_metric ({len(train_metric)}), test_metric ({len(test_metric)})")

            plt.figure(figsize=(14, 10))
            plt.plot(epochs_range, train_metric, label=f'Training {metric_name}')
            plt.plot(epochs_range, test_metric, label=f'Testing {metric_name}')
            plt.xlabel('Epochs')
            plt.ylabel(ylabel)



            # Determine if we are looking for max or min and set the appropriate title
            if 'Loss' in metric_name:
                min_loss_epoch, min_loss_value = min(enumerate(train_metric), key=lambda x: x[1])
                annotate_text = f"{min_loss_value:.4f}"
                # Adjust the position of the annotation text
                text_position = (min_loss_epoch, min_loss_value * 1.05)  # slightly above the actual point
                arrowprops_dict = dict(facecolor='red', arrowstyle='wedge,tail_width=0.3', shrinkA=10, shrinkB=5, linewidth=1.5)
            else:
                max_acc_epoch, max_acc_value = max(enumerate(train_metric), key=lambda x: x[1])
                annotate_text = f"{max_acc_value:.2f}%"
                # Adjust the position of the annotation text
                text_position = (max_acc_epoch, max_acc_value * 0.95)  # slightly below the actual point
                arrowprops_dict = dict(facecolor='green', arrowstyle='wedge,tail_width=0.3', shrinkA=10, shrinkB=5, linewidth=1.5)

            # Annotate the plot with the highest/lowest value
            plt.annotate(annotate_text,
                        xy=(max_acc_epoch, max_acc_value) if 'Accuracy' in metric_name else (min_loss_epoch, min_loss_value),
                        xytext=text_position,
                        arrowprops=arrowprops_dict,
                        fontsize=12)

            plt.title(f'Training and Testing {metric_name} over Epochs - {plot_id}')
            # plt.legend()
            plt.legend(loc='upper left')
            plt.savefig(f"{self.log_save_path}_{metric_name}_{plot_id}.png")
            # plt.show()

        plot_generic_metrics(epochs_range, train_losses, test_losses, 'Loss', 'Loss')
        plot_generic_metrics(epochs_range, train_accuracies, test_accuracies, 'Accuracy', 'Accuracy (%)')

