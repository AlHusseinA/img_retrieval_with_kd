from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np
import torch

# Extract features for the entire dataset
# Assuming `data_loader` is your DataLoader for the dataset whether traindataloader or testdataloader


def pca_fit_features(model, data_loader, n_components=200, whiten=False, device=None):
    
    vanilla_feature_size = 2048
    
    # Asserts and assumptions
    assert isinstance(model, torch.nn.Module), "Model must be an instance of torch.nn.Module"
    assert isinstance(data_loader, DataLoader), "Data loader must be an instance of torch.utils.data.DataLoader"    
    assert n_components > 0, "n_components must be a positive integer"

    features_list = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()  # Set the model to evaluation mode

    for inputs, _ in data_loader:
        with torch.inference_mode():
            inputs = inputs.to(device)
            features = model(inputs)
            features = features.view(features.size(0), -1)  # Flatten the features
            features_list.append(features.cpu().data.numpy())
            # features_list.append(feature.cpu())  # Move to CPU before appending
            # print(f"Current batch features shape: {features.cpu().data.numpy().shape}")  # Debugging


    # Concatenate all features into a single numpy array

    features = np.concatenate(features_list, axis=0)
    print(f"Shape of features after concatenation: {features.shape}")
    if np.isnan(features).any() or np.isinf(features).any():
        raise ValueError("Features contain NaN or infinite values")


    # Ensure n_components is valid
    assert n_components <= features.shape[1], "n_components cannot be greater than the number of features"
    assert features.shape[1] == vanilla_feature_size, "Feature size must match expected vanilla feature size"

    # Apply PCA
    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(features)
    
    return pca, features



