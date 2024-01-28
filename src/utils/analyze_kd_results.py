import os
import json
import re
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def analyze_metrics(directory_path, T):
    results = {}

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            # Extract student_size from the filename
            match = re.search(r"student_size_(\d+)", filename)
            if match:
                student_size = int(match.group(1))
            else:
                continue  # Skip file if student_size not found

            json_file_path = os.path.join(directory_path, filename)
            with open(json_file_path, 'r') as file:
                data = json.load(file)

            # Extract metrics
            highest_train_acc = max(data['train_acc'])
            last_train_acc = data['train_acc'][-1]
            highest_val_acc = max(data['val_acc'])
            last_val_acc = data['val_acc'][-1]
            highest_train_loss = min(data['train_loss'])
            last_train_loss = data['train_loss'][-1]
            highest_val_loss = min(data['val_loss'])
            last_val_loss = data['val_loss'][-1]

            # Store results in the dictionary
            results[student_size] = {
                'highest_train_acc': highest_train_acc,
                'last_train_acc': last_train_acc,
                'highest_val_acc': highest_val_acc,
                'last_val_acc': last_val_acc,
                'highest_train_loss': highest_train_loss,
                'last_train_loss': last_train_loss,
                'highest_val_loss': highest_val_loss,
                'last_val_loss': last_val_loss
            }

            # Print results neatly
            print(f"\nResults for Temperature: {T}\n")
            print(f"\nStudent Size: {student_size}")

            print(f"  - Highest Val Accuracy:   {highest_val_acc:.4f}")
            print(f"  - Last Val Accuracy:      {last_val_acc:.4f}")
            print(f"###"*25)
            print(f"  - Highest Train Accuracy: {highest_train_acc:.4f}")
            print(f"  - Last Train Accuracy:    {last_train_acc:.4f}")            
            print(f"  - Highest Train Loss:     {highest_train_loss:.4f}")
            print(f"  - Last Train Loss:        {last_train_loss:.4f}")
            print(f"  - Highest Val Loss:       {highest_val_loss:.4f}")
            print(f"  - Last Val Loss:          {last_val_loss:.4f}")

    return results



def plot_val_acc_vs_model_size(results, directory_path, T):
    # Prepare data for plotting
    # Sort the keys to ensure the line goes from left to right
    model_sizes = sorted(results.keys())
    val_accs = [results[size]['highest_val_acc'] for size in model_sizes]

    # Create a large figure
    plt.figure(figsize=(12, 8))

    # Scatter plot of val_acc vs model size
    plt.scatter(model_sizes, val_accs, color='blue')

    # Connect dots with a dashed line, ensuring it's sequential
    plt.plot(model_sizes, val_accs, '--', color='red')

    # Setting the x-axis to only contain discrete student sizes
    plt.xticks(model_sizes)

    # Setting the y-axis range from 20 to 100
    plt.ylim(20, 100)

    # Labels and title
    plt.xlabel('Model Size (Student Size)')
    plt.ylabel('Highest Validation Accuracy')
    plt.title(f'Relationship Between Validation Accuracy and Model Size at Temperature = {T}')

    # Save the figure
    plt.savefig(f"{directory_path}/val_acc_vs_model_size_T_{T}.png")




# def create_plots(size, max_val, temperature, directory_path):
#     # Plot 1: Relationship between max_val and size
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x=size, y=max_val)
#     plt.xlabel('Size')
#     plt.ylabel('Max Validation Accuracy')
#     plt.title('Max Validation Accuracy vs Size')
#     plt.savefig(f"{directory_path}/max_val_vs_size.png")
#     plt.close()

#     # Plot 2: Relationship between max_val and temperature
#     plt.figure(figsize=(10, 6))
#     sns.scatterplot(x=temperature, y=max_val)
#     plt.xlabel('Temperature')
#     plt.ylabel('Max Validation Accuracy')
#     plt.title('Max Validation Accuracy vs Temperature')
#     plt.savefig(f"{directory_path}/max_val_vs_temperature.png")
#     plt.close()

#     # Plot 3: 3D plot of Size, Temperature, and Max Validation Accuracy
#     fig = plt.figure(figsize=(12, 8))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.scatter(size, temperature, max_val, c=max_val, cmap='viridis', marker='o')
#     ax.set_xlabel('Size')
#     ax.set_ylabel('Temperature')
#     ax.set_zlabel('Max Validation Accuracy')
#     ax.set_title('3D Plot of Size, Temperature, and Max Validation Accuracy')
#     plt.savefig(f"{directory_path}/3d_plot_size_temp_max_val.png")
#     plt.close()


def plot_max_val_vs_temperature(size, max_vals, temperature, directory_path):
    if len(max_vals) != len(temperature):
        raise ValueError("The length of max_vals and temperature lists must be the same.")

    # Create a figure
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=temperature, y=max_vals)
    plt.xlabel('Temperature')
    plt.ylabel('Max Validation Accuracy')
    plt.title(f'Max Validation Accuracy vs Temperature for Model Size {size}')

    # Save the figure in the specified directory
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)  # Create directory if it does not exist
    file_name = f"max_val_vs_temp_size_{size}.png"
    plt.savefig(os.path.join(directory_path, file_name))
    

# Example usage of the function

# Example usage
dir = "/home/alabutaleb/Desktop/confirmation/kd_logs/logs_temperature_0.07999999999999999_sizes_[8, 16, 32]"
# dir = "/home/alabutaleb/Desktop/confirmation/kd_logs/logs_temperature_0.08999999999999998_sizes_[8, 16, 32]"
# dir = "/home/alabutaleb/Desktop/confirmation/kd_logs/logs_temperature_0.09999999999999998_sizes_[8, 16, 32]"
# dir = "/home/alabutaleb/Desktop/confirmation/kd_logs/logs_temperature_0.06999999999999999_sizes_[8, 16, 32]"
# dir = "/home/alabutaleb/Desktop/confirmation/kd_logs/logs_temperature_0.06_sizes_[8, 16, 32]"
# dir = "/home/alabutaleb/Desktop/confirmation/kd_logs/best_logs_temperature_0.085_sizes_[8, 32, 128, 512]"
T=0.08
metrics_results = analyze_metrics(dir, T)
plot_val_acc_vs_model_size(metrics_results, dir, T)

dir_save_temperaturevsmaxval = "/home/alabutaleb/Desktop/confirmation/kd_logs/plots_temperature_vs_max_val"
max_vals_8 = [70.7801, 71.4360, 74.4563, 70.5040,70.8837, 72.4025]
temperature = [0.06, 0.07, 0.08, 0.085, 0.09, 0.1]
max_vals_16 = [80.3072, 79.8067, 80.6869, 0,80.2554, 80.5316]
size=16
plot_max_val_vs_temperature(size, max_vals_16, temperature, dir_save_temperaturevsmaxval)

# size_list = [8, 16, 32, 128, 512]
# create_plots(size_list, max_val_list, temperature_list, 'path_to_directory')

