

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """ Denormalizes image tensors using the provided mean and std """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)  # The inverse of normalization
    return tensor

def normalize_image(image):
    """ Normalize images to [0, 1] range """
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image

def load_images_from_indices(data_loader, indices):
    images = []
    for idx in indices:
        image, _ = data_loader.dataset[idx]
        image = denormalize(image)  # Denormalize the image
        images.append(image)
    return torch.stack(images)

def visualize_retrieval_first_n_samples(query_idx, retrieval_indices, data_loader, save_dir, label_to_name):
    query_image = load_images_from_indices(data_loader, [query_idx])
    retrieval_images = load_images_from_indices(data_loader, retrieval_indices)

    # Normalize for display
    query_image = normalize_image(query_image)
    retrieval_images = normalize_image(retrieval_images)

    # Combine query and retrieved images
    combined_images = torch.cat((query_image, retrieval_images), dim=0)
    grid = make_grid(combined_images, nrow=11)

    # Create the figure
    fig, ax = plt.subplots(figsize=(28, 10))
    ax.imshow(grid.permute(1, 2, 0))
    ax.axis('off')

    # Annotations
    grid_width = grid.size(2)
    single_image_width = grid_width / 11
    start_x_position = single_image_width / 2

    # Ensure that all indices are being processed
    all_indices = [query_idx] + retrieval_indices
    # print("All indices being processed:", all_indices)

    for i, idx in enumerate(all_indices):
        label = data_loader.dataset.targets[idx]
        full_class_name = label_to_name[label]
        class_name = full_class_name[4:] if len(full_class_name) > 4 else full_class_name

        text_x_position = start_x_position + i * single_image_width
        ax.text(text_x_position, grid.size(1) + 10, class_name, ha='center', va='top')

    plt.title(f'Query {query_idx} and Top 10 Retrievals')
    plt.savefig(f'{save_dir}/query_{query_idx}.png')
    plt.close()



# def visualize_retrieval_first_n_samples(query_idx, retrieval_indices, data_loader, save_dir):
#     query_image = load_images_from_indices(data_loader, [query_idx])
#     retrieval_images = load_images_from_indices(data_loader, retrieval_indices)

#     # Normalize for display
#     query_image = normalize_image(query_image)
#     retrieval_images = normalize_image(retrieval_images)

#     # Combine query and retrieved images
#     combined_images = torch.cat((query_image, retrieval_images), dim=0)
#     grid = make_grid(combined_images, nrow=11)

#     plt.figure(figsize=(20, 5))
#     plt.imshow(grid.permute(1, 2, 0))
#     plt.axis('off')
#     plt.title(f'Query {query_idx} and Top 10 Retrievals')
#     plt.savefig(f'{save_dir}/query_{query_idx}.png')
#     plt.close()









# # import os
# # import matplotlib.pyplot as plt

# # def visualize_retrieval_first_n_samples(query_image, retrieved_indices, gallery_dataset, dir_results_visuals, n_retrieved=10, query_index=0):
# #     """
# #     Visualizes the query image and its top retrieved images, and saves the figure to a specified directory.

# #     Parameters:
# #     query_image (Tensor): The query image tensor.
# #     retrieved_indices (Tensor): Indices of the retrieved images from the gallery.
# #     gallery_dataset (Dataset): The dataset from which to retrieve gallery images.
# #     dir (str): Directory path to save the visualizations.
# #     n_retrieved (int): Number of top retrieved images to display.
# #     query_index (int): Index of the query image, used for naming the saved file.
# #     """
# #     # Ensure the directory exists
# #     if not os.path.exists(dir_results_visuals):
# #         os.makedirs(dir_results_visuals)

# #     # Define figure size
# #     fig_size = (28, 10)

# #     fig, axes = plt.subplots(1, n_retrieved + 1, figsize=fig_size)

# #     # Display the query image
# #     try:
# #         axes[0].imshow(query_image.permute(1, 2, 0))
# #         axes[0].set_title("Query Image")
# #         axes[0].axis('off')
# #     except Exception as e:
# #         print(f"Error displaying query image: {e}")
# #         return

# #     # Display top retrieved images
# #     for i, idx in enumerate(retrieved_indices[:n_retrieved]):
# #         try:
# #             img = gallery_dataset[idx][0].permute(1, 2, 0)
# #             axes[i + 1].imshow(img)
# #             axes[i + 1].set_title(f"Rank {i+1}")
# #             axes[i + 1].axis('off')
# #         except Exception as e:
# #             print(f"Error displaying image at index {idx}: {e}")

# #     # Save the figure with a simple filename
# #     plt.savefig(os.path.join(dir, f'query_{query_index}.png'))
# #     plt.close(fig)

# import os
# import matplotlib.pyplot as plt

# def visualize_retrieval_first_n_samples(query_image, retrieved_indices, gallery_dataset, testloader, dir_results_visuals, 
#                         n_retrieved=10, query_index=0, 
#                         image_extractor=lambda dataset, idx: dataset[idx][0]):
#     """
#     Visualizes the original query image and its top retrieved images (original images), and saves the figure to a specified dir_results_visualsectory.

#     Parameters:
#     query_image (Tensor): The feature tensor of the query image (used for matching).
#     retrieved_indices (Tensor): Indices of the retrieved images from the gallery.
#     gallery_dataset (Dataset): The dataset from which to retrieve gallery images (features).
#     testloader (DataLoader): The DataLoader containing the original images.
#     dir_results_visuals (str): dir_results_visualsectory path to save the visualizations.
#     n_retrieved (int): Number of top retrieved images to display.
#     query_index (int): Index of the query image, used for naming the saved file.
#     image_extractor (function): A function that takes the dataset and an index, and returns the corresponding original image.
#     """
#     # Define figure size inside the function
#     fig_size = (28, 10)
#     dir_res = os.path.join(dir_results_visuals, f'query_{query_index}.png')
#     # Ensure the directory exists
#     if not os.path.exists(dir_res):
#         os.makedirs(dir_res)

#     fig, axes = plt.subplots(1, n_retrieved + 1, figsize=fig_size)

#     # Display the original query image
#     try:
#         # Assuming the query image index is used to fetch the corresponding image from the testloader
#         original_query_image = image_extractor(testloader, query_index)
#         if original_query_image.dim() == 3:
#             axes[0].imshow(original_query_image.permute(1, 2, 0))
#         else:
#             raise ValueError("Original query image does not have 3 dimensions")
#         axes[0].set_title("Query Image")
#         axes[0].axis('off')
#     except Exception as e:
#         print(f"Error displaying original query image: {e}")
#         return

#     # Display top retrieved original images
#     for i, idx in enumerate(retrieved_indices[:n_retrieved]):
#         try:
#             # Fetching original image corresponding to the feature index from the testloader
#             img = image_extractor(testloader, idx)
#             if img.dim() == 3:
#                 axes[i + 1].imshow(img.permute(1, 2, 0))
#             else:
#                 raise ValueError(f"Original image at index {idx} does not have 3 dimensions")
#             axes[i + 1].set_title(f"Rank {i+1}")
#             axes[i + 1].axis('off')
#         except Exception as e:
#             print(f"Error displaying original image at index {idx}: {e}")

#     # Save the figure with a simple filename
#     plt.savefig(dir_res)
#     plt.close(fig)


