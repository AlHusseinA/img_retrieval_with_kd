
import matplotlib.pyplot as plt
from itertools import cycle
import json
import numpy as np
import os
import re

def plot_multiple_metrics(log_save_path):# , feature_size, lr, dataset_name):
    # Create a cycle iterator for colors
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    folder_path = f"{log_save_path}"
    
    # Initialize plots
    plt.figure(figsize=(40, 24))
    
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
    plt.legend(loc='upper right')
    plt.subplot(1, 2, 2)
    plt.legend(loc='upper right')
    # save figure
    plt.savefig(f"{log_save_path}/metrics_plot.png")

    # plt.tight_layout()
    # plt.show()

path = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiments Wed 1st Nov 23 linear not conv compression/logs_new"
lr = 1e-05


plot_multiple_metrics(path)#, feature_size, lr, "cub200")


def plot_multiple_metrics2(log_save_path):
    # Create a cycle iterator for colors
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    folder_path = f"{log_save_path}"
    
    # Increase figure size for better readability
    plt.figure(figsize=(40, 24))  # Increased width from 32 to 40
    
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
            
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            epochs = data['epochs']
            train_acc = data['train_acc']
            val_acc = data['val_acc']
            train_loss = data['train_loss']
            val_loss = data['val_loss']
            
            # Plot Training and Validation Accuracy with annotations
            plt.subplot(1, 2, 1)
            # train_acc_line, = plt.plot(epochs, train_acc, color=color)
            train_acc_line = plt.plot(epochs, train_acc, color=color)[0]
            
            
            # val_acc_line, = plt.plot(epochs, val_acc, '--', color=color)
            val_acc_line = plt.plot(epochs, val_acc, '--', color=color)[0]
            # Annotate the lines instead of using a legend
            # plt.text(epochs[-1], train_acc[-1], f'{filename} - Train', color=train_acc_line.get_color())
            # plt.text(epochs[-1], val_acc[-1], f'{filename} - Val', color=val_acc_line.get_color())
            # Annotate the lines instead of using a legend
            plt.text(epochs[-1], train_acc[-1], f'{filename} - Train', color=train_acc_line.get_color())
            plt.text(epochs[-1], val_acc[-1], f'{filename} - Val', color=val_acc_line.get_color())


            # Plot Training and Validation Loss
            plt.subplot(1, 2, 2)
            plt.plot(epochs, train_loss, label=f'Training Loss: {filename}', color=color)
            plt.plot(epochs, val_loss, '--', label=f'Validation Loss: {filename}', color=color)
    
    # Show legend only for loss subplot
    plt.subplot(1, 2, 2)
    plt.legend(loc='upper right')
    
    # Save the figure
    plt.savefig(f"{log_save_path}/metrics_plot_wide2.png")
    
    # Uncomment the following lines if you want to display the plot as well
    # plt.tight_layout()
    # plt.show()

path = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiments Wed 1st Nov 23 linear not conv compression/logs_new"
lr = 1e-05


plot_multiple_metrics2(path)#, feature_size, lr, "cub200")


# def plot_metrics_with_annotations(log_save_path):
#     # Create a cycle iterator for colors
#     colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    
#     # Initialize plots
#     plt.figure(figsize=(32, 24))
    
#     # Subplot for accuracy
#     accuracy_ax = plt.subplot(1, 2, 1)
#     plt.xlabel('Epochs')
#     plt.ylabel('Accuracy')
#     plt.title('Training and Validation Accuracy')
    
#     # Subplot for loss
#     loss_ax = plt.subplot(1, 2, 2)
#     plt.xlabel('Epochs')
#     plt.ylabel('Loss')
#     plt.title('Training and Validation Loss')
    
#     # Regex pattern to extract feature size from the file name
#     pattern = re.compile(r"feature_size_(\d+)_CUB-200\.json")
    
#     # Loop over each JSON file in the folder to read and plot data
#     for filename in os.listdir(log_save_path):
#         if filename.endswith(".json"):
#             json_file_path = os.path.join(log_save_path, filename)
#             color = next(colors)  # Get the next color in the cycle
            
#             # Extract feature size from the filename using regex
#             match = pattern.search(filename)
#             if match:
#                 feature_size = match.group(1)
#             else:
#                 continue  # If the pattern does not match, skip the file
            
#             # Read the JSON file into a Python dictionary
#             with open(json_file_path, 'r') as file:
#                 data = json.load(file)
            
#             # Extract the metrics
#             epochs = data['epochs']
#             train_acc = data['train_acc']
#             val_acc = data['val_acc']
#             train_loss = data['train_loss']
#             val_loss = data['val_loss']
            
#             # Plot Training and Validation Accuracy with annotation
#             accuracy_ax.plot(epochs, train_acc, label=f'Feature Size: {feature_size}', color=color)
#             accuracy_ax.plot(epochs, val_acc, '--', color=color)
#             accuracy_ax.annotate(f'{feature_size}', xy=(epochs[-1], train_acc[-1]), xytext=(10,0),
#                                  textcoords='offset points', va='center', color=color)
            
#             # Plot Training and Validation Loss with annotation
#             loss_ax.plot(epochs, train_loss, color=color)
#             loss_ax.plot(epochs, val_loss, '--', color=color)
#             loss_ax.annotate(f'{feature_size}', xy=(epochs[-1], train_loss[-1]), xytext=(10,0),
#                              textcoords='offset points', va='center', color=color)
    
#     # Show legends
#     # accuracy_ax.legend()
#     # loss_ax.legend()
#     accuracy_ax.legend(loc='upper right')
#     loss_ax.legend(loc='upper right')
#     # plt.tight_layout()
#     plt.savefig(f"{log_save_path}/metrics_plot3.png")


# path = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiments Wed 1st Nov 23 linear not conv compression/logs_new"
# lr = 1e-05


# plot_metrics_with_annotations(path)#, feature_size, lr, "cub200")



