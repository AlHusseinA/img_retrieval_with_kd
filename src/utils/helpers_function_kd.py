

import os
import json
from itertools import cycle
import matplotlib.pyplot as plt

def plot_performance(log_save_path, T, student_size=None):
    # Create log_plots directory if it doesn't exist
    plots_dir = os.path.join(log_save_path, 'log_plots_all_jsons')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # # Print log_save_path for debugging
    # print(f"####" * 50)
    # print(f"Plotting performance for files in: {log_save_path}")
    # print(f"####" * 50)

    # Create a cycle iterator for colors
    colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])

    # Initialize accuracy plot
    plt.figure(figsize=(22, 18))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    # Loop over each JSON file in the folder to read and plot accuracy data
    for filename in os.listdir(log_save_path):
        if filename.endswith(".json"):
            json_file_path = os.path.join(log_save_path, filename)
            color = next(colors)  # Get the next color in the cycle

            # Read the JSON file into a Python dictionary
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            if not data['epochs']:  # Check if epochs data is empty
                continue  # Skip plotting for this file if epochs data is empty

            epochs = data['epochs']
            train_acc = data['train_acc']
            val_acc = data['val_acc']

            # Plot Training and Validation Accuracy
            plt.plot(epochs, train_acc, label=f'Training Accuracy: {filename}', color=color)
            plt.plot(epochs, val_acc, '--', label=f'Validation Accuracy: {filename}', color=color)

    plt.legend()
    # Define and save accuracy plot filename
    accuracy_filename = f'accuracy_plots_all_students_temperature_{T}.png'
    plt.savefig(os.path.join(plots_dir, accuracy_filename))
    plt.close()

    # Initialize loss plot
    plt.figure(figsize=(22, 18))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    # Loop over each JSON file in the folder to read and plot loss data
    for filename in os.listdir(log_save_path):
        if filename.endswith(".json"):
            json_file_path = os.path.join(log_save_path, filename)
            color = next(colors)

            with open(json_file_path, 'r') as file:
                data = json.load(file)

            if not data['epochs']:  # Check if epochs data is empty
                continue  # Skip plotting for this file if epochs data is empty

            epochs = data['epochs']
            train_loss = data['train_loss']
            val_loss = data['val_loss']

            # Plot Training and Validation Loss with epochs as x-axis
            plt.plot(epochs, train_loss, label=f'Training Loss: {filename}', color=color)
            plt.plot(epochs, val_loss, '--', label=f'Validation Loss: {filename}', color=color)

    plt.legend()
    # Define and save loss plot filename
    loss_filename = f'loss_plots_plot_all_students_temperature_{T}.png'
    plt.savefig(os.path.join(plots_dir, loss_filename))
    plt.close()

# Example usage
# plot_performance('/path/to/log/directory')



def plot_single_performance(json_file_path):
    # Extract directory from file path and create log_plots directory if it doesn't exist
    log_save_path = os.path.dirname(json_file_path)
    plots_dir = os.path.join(log_save_path, 'log_plots')
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    # Check if the specified file is a JSON file
    if not json_file_path.endswith(".json"):
        print("The specified file is not a JSON file.")
        return

    # Initialize accuracy plot
    plt.figure(figsize=(22, 18))
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')

    # Read the JSON file into a Python dictionary
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    if not data['epochs']:  # Check if epochs data is empty
        print("No data found in the file.")
        return

    epochs = data['epochs']
    train_acc = data['train_acc']
    val_acc = data['val_acc']

    # Plot Training and Validation Accuracy
    plt.plot(epochs, train_acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, '--', label='Validation Accuracy')
    plt.legend()

    # Define and save accuracy plot filename
    accuracy_filename = os.path.basename(json_file_path).replace('.json', '_accuracy_plot.png')
    plt.savefig(os.path.join(plots_dir, accuracy_filename))
    plt.close()

    # Initialize loss plot
    plt.figure(figsize=(22, 18))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')

    train_loss = data['train_loss']
    val_loss = data['val_loss']

    # Plot Training and Validation Loss with epochs as x-axis
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, '--', label='Validation Loss')
    plt.legend()

    # Define and save loss plot filename
    loss_filename = os.path.basename(json_file_path).replace('.json', '_loss_plot.png')
    plt.savefig(os.path.join(plots_dir, loss_filename))
    plt.close()

# Example usage
# plot_single_performance('/path/to/specific/file.json')