import matplotlib.pyplot as plt
from itertools import cycle
import json
import numpy as np
from typing import Dict
import os

# from utils.helper_functions import plot_multiple_metrics
# plot_multiple_metrics
# plt.savefig(f"{self.log_save_path}_{metric_name}_{plot_id}.png")

class MetricsLogger:
    def __init__(self, filepath="./logs/metrics.json"):

        # self.metrics_dict = {dataset_name: {"loss": [], "mAP": [], "R@1": [], "R@5": [], "R@10": []} 
        #                      for dataset_name in dataset_names}
        self.data = {}  # A nested dictionary to hold all metrics
        self.filepath = filepath
        # self.dataset_names = dataset_names


    
    def plot_multiple_metrics(self,log_save_path, feature_size, lr, dataset_name):
        # Create a cycle iterator for colors
        colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
        folder_path = f"{log_save_path}"
        
        # Initialize plots
        plt.figure(figsize=(18, 6))
        
        # Subplot for accuracy
        plt.subplot(1, 2, 1)
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.title('Training and Validation Accuracy')
        
        # Subplot for loss
        plt.subplot(1, 2, 2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        
        # Loop over each JSON file in the folder to read and plot data
        for filename in os.listdir(folder_path):
            if filename.endswith(".json"):
                json_file_path = os.path.join(folder_path, filename)
                color = next(colors)  # Get the next color in the cycle
                
                # Step 1: Read the JSON file into a Python dictionary
                with open(json_file_path, 'r') as file:
                    data = json.load(file)
                
                # Step 2: Data Preparation
                epochs = data['epochs']
                train_acc = data['train_acc']
                val_acc = data['val_acc']
                train_loss = data['train_loss']
                val_loss = data['val_loss']
                
                # Step 3: Plotting
                
                # Plot Training and Validation Accuracy
                plt.subplot(1, 2, 1)
                plt.plot(epochs, train_acc, label=f'Training Accuracy: {filename}', color=color)
                plt.plot(epochs, val_acc, '--', label=f'Validation Accuracy: {filename}', color=color)
                
                # Plot Training and Validation Loss
                plt.subplot(1, 2, 2)
                plt.plot(epochs, train_loss, label=f'Training Loss: {filename}', color=color)
                plt.plot(epochs, val_loss, '--', label=f'Validation Loss: {filename}', color=color)
        
        # Show legends
        plt.subplot(1, 2, 1)
        plt.legend(loc='upper left')
        plt.subplot(1, 2, 2)
        plt.legend(loc='upper left')
        
        plt.tight_layout()
        plt.show()




    def log_metrics(self, feature_size, epoch, training_loss, train_acc, avg_val_loss, val_acc, dataset_name, log_save_path,lr):
    # def log_metrics(self, feature_size, epoch, training_loss, train_acc, avg_val_loss, val_acc, batch_accuracies, dataset_name):

        if feature_size not in self.data:
            self.data[feature_size] = {'epochs': [], 'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []} #, 'batch_accuracies': []}
        
        self.data[feature_size]['epochs'].append(epoch)
        self.data[feature_size]['train_loss'].append(training_loss)
        self.data[feature_size]['val_loss'].append(avg_val_loss)
        self.data[feature_size]['train_acc'].append(train_acc)
        self.data[feature_size]['val_acc'].append(val_acc)
        # self.data[feature_size]['batch_accuracies'].append(batch_accuracies)
        # Save to JSON, using feature size and dataset name as part of the file name
        
        # json_file_path = f"./logs/classification_metrics_{feature_size}_{dataset_name}.json"
        json_file_path = f"{log_save_path}/classification_metrics_lr_{lr}_feature_size_{feature_size}_{dataset_name}.json"

        with open(json_file_path, 'w') as file:
            json.dump(self.data[feature_size], file)

    def log_retrieval_metrics(self, feature_size, metrics, dataset_name):
        # Consistency check
        # print(f"metrics.keys(): {metrics.keys()}")
        # print(f"set(metrics.keys()): {set(metrics.keys())}")
        # print(f"metrics: {metrics}")
        assert set(metrics.keys()) == {'mAP', 'R@1', 'R@5', 'R@10'}, "Unexpected metrics keys"

        # Initialize data structure
        if feature_size not in self.data:
            self.data[feature_size] = {}
        
        if 'retrieval' not in self.data[feature_size]:
            self.data[feature_size]['retrieval'] = {'mAP': [], 'R@1': [], 'R@5': [], 'R@10': []}

        # Log metrics
        for key, value in metrics.items():
            self.data[feature_size]['retrieval'][key].append(value)  # Convert tensor to Python number

        # Create logs directory if it doesn't exist
        if not os.path.exists("./logs"):
            os.makedirs("./logs")

        # Save to JSON
        json_file_path = f"./logs/retrieval_metrics_{feature_size}_{dataset_name}.json"
        with open(json_file_path, 'w') as file:
            json.dump(self.data[feature_size]['retrieval'], file)

    def plot_metrics_by_epoch(self, feature_size):
        plot_id = f"{self.dataset_names}_ResNet50_FeatureSize_{self.feature_size}"

        epochs = self.data[feature_size]['epochs']
        plt.figure()
        plt.plot(epochs, self.data[feature_size]['train_loss'], label='Train Loss')
        plt.plot(epochs, self.data[feature_size]['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title(f'Metrics by Epoch for feature size {feature_size}')
        plt.show()
        plt.savefig(f"Loss_{plot_id}_from_logger.png")
        self.plot_metrics_by_feature_size()
        self.save_metrics(self.filepath)


    def plot_metrics_by_feature_size(self, metric='val_loss'):
        plot_id = f"{self.dataset_names}_ResNet50_FeatureSize_{self.feature_size}"

        feature_sizes = list(self.data.keys())
        metric_values = [self.data[fs][metric][-1] for fs in feature_sizes]  # Take the last recorded metric
        plt.figure()
        plt.plot(feature_sizes, metric_values, marker='o')
        plt.xlabel('Feature Size')
        plt.ylabel(metric)
        plt.title(f'{metric} by Feature Size')
        plt.show()       
        plt.savefig(f"{metric}_{plot_id}_from_logger.png")   

  