import torch
import numpy as np


def generate_and_process_features(features_extractor, dataloader, device):
    features_extractor.eval()  # Set the feature extractor to evaluation mode
    features_list = []

    with torch.inference_mode():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            features = features_extractor(inputs)
            features_list.append(features.cpu().data.numpy())
            # features_list.append(features.cpu())
            # processed_features = f(features)
            # processed_features_list.append(processed_features.cpu().data.numpy())
    # Save processed features to a file or database
    # ...
    return features_list


# def batch_features(features, batch_size):
#     # Split features into batches
#     num_batches = features.shape[0] // batch_size
#     return np.array_split(features, num_batches)

def batch_features(features, batch_size):
    # Split features into batches with the same logic as DataLoader
    return [features[i:i + batch_size] for i in range(0, len(features), batch_size)]


def make_predictions_model(model, features_list, device):
    '''This function will generate predictions from either feature extractor backbone or the new
    independant fc layer'''
    # inputs can be either compressed features or images
    # Add assert unit tests to:
    # match the dims of the one feature vector with the input to the model

    model.eval()
    # new_fc.eval()  # Set the new fc layer to evaluation mode
    predictions = []

    with torch.inference_mode():
        for features in features_list:
            features = torch.tensor(features).to(device)
            output = model(features)
            # output = new_fc(features)
            predictions.append(output.cpu().data.numpy())

    return predictions
