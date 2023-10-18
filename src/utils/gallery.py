import torch
from tqdm import tqdm
import numpy as np

def extract_gallery_features(model, trainloader, device):
    """
    This function extracts features for all the gallery images using the provided model.

    Args:
    - model: The trained model to use for feature extraction.
    - trainloader: DataLoader for the training set (gallery set).
    - device: The device (cpu or cuda) to use for computations.

    Returns:
    - gallery_features: A numpy array of extracted features for the gallery images.
    - gallery_labels: A numpy array of labels for the gallery images.
    """
    model.eval()

    # Initialize lists to store gallery features and labels
    gallery_features = []
    gallery_labels = []

    for images, labels in tqdm(trainloader):
        images, labels = images.to(device), labels.to(device)
        with torch.inference_mode():  # Use torch.no_grad() instead of torch.inference_mode()
            features = model(images)
            gallery_features.append(features)
            gallery_labels.append(labels)

    # Concatenate all features and labels into numpy arrays
    gallery_features = torch.cat(gallery_features).cpu().numpy()
    gallery_labels = torch.cat(gallery_labels).cpu().numpy()

    return gallery_features, gallery_labels


# Comments:
# Currently, the function signature is defined but the body of the function is empty. It needs to be implemented to suit the functionality it is intended for.
# The function takes three parameters: the model, the dataloader, and the device, which are appropriate inputs to facilitate the extraction of gallery features.
# Suggested Adjustments:
# Implement the function to iterate over all the batches in the dataloader, use the model to get the features for each batch, and collect these features and the corresponding labels in lists. This would involve setting the model to evaluation mode and utilizing a with torch.no_grad(): block to ensure gradients are not calculated during this process.
# After collecting the features and labels in lists, they should be concatenated to get numpy arrays which can be used further in the pipeline.
# This function will be a critical part of your pipeline, facilitating the extraction of features from your dataset which will be used in the retrieval process. It's important to ensure that it functions correctly and efficiently, potentially using batch processing to speed up the feature extraction process.

