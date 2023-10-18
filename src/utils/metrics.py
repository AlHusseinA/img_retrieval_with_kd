import numpy as np
from typing import List, Tuple
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder

def calculate_map(predictions: np.ndarray, ground_truths: np.ndarray) -> float:
    """
        Calculate the mean Average Precision (mAP).
        
        Parameters:
        - predictions (numpy.ndarray): Array containing prediction scores.
        - ground_truths (numpy.ndarray): Array containing ground truth labels.

        Returns:
        - float: The mean average precision score.
    """
    assert predictions.shape == ground_truths.shape, "Shape mismatch"

    # predictions are the sorted scores from image_retrieval function
    # ground_truths are the labels of the gallery images
    print(f"&^&"*20)
    print("You are now in calculate_map function")
    print(f"Shape of predictions: {predictions.shape}")
    print(f"Shape of ground_truths: {ground_truths.shape}")
    print(f"Type of predictions: {type(predictions)}")
    print(f"Type of ground_truths: {type(ground_truths)}")
    print(f"&^&"*20)
    # if not isinstance(predictions, np.ndarray) or not isinstance(ground_truths, np.ndarray):
    #     raise TypeError("Both predictions and ground_truths must be numpy ndarrays")

    # Error checking for dimensions
    if predictions.shape[0] != ground_truths.shape[0]:
        raise ValueError("The number of predictions and ground truths should be the same.")
        
    if len(predictions.shape) != 1 or len(ground_truths.shape) != 1:
        raise ValueError("Both predictions and ground_truths should be 1D arrays.")
    
    # if ground_truths.is_cuda:
    #     ground_truths = ground_truths.cpu()

    ground_truths = ground_truths.cpu().numpy()
    predictions = predictions.cpu().numpy()
    encoder = OneHotEncoder(sparse=False, categories='auto')
    # ground_truths_one_hot = encoder.fit_transform(ground_truths.numpy().reshape(-1, 1))
    ground_truths_one_hot = encoder.fit_transform(ground_truths.reshape(-1, 1))
    print("XFX"*20)
    print(f"type of ground_truths_one_hot: {type(ground_truths_one_hot)}")
    print(f"shape of ground_truths_one_hot: {ground_truths_one_hot.shape}")
    print(f"tpe of predictions: {type(predictions)}")
    # print(f"Shape of ground_truths_one_hot[i]: {ground_truths_one_hot[i].shape}")
    # print(f"Shape of predictions[i]: {predictions[i].shape}")
    print(f"{predictions[:5]=}")
    # ground_truths_one_hot = encoder.fit_transform(ground_truths.reshape(-1, 1))
    cumulative_ap = 0.0
    # print(f"Shape of ground_truths_one_hot: {ground_truths_one_hot.shape}")
    # print(f"Shape of predictions: {predictions.shape}")
    for i in range(len(ground_truths_one_hot)):

        if np.sum(ground_truths[i]) == 0:
            ap = 0
        else:
            print(f"Shape of predictions[i]: {predictions[i].shape}")
            ap = average_precision_score(ground_truths_one_hot[i], predictions[i])
            
        cumulative_ap += ap
    
    map_score = cumulative_ap / len(predictions)
    return map_score





def calculate_recall_at_k(predictions, ground_truths, k):
    """
    Calculate the recall at k.

    Parameters:
    - predictions (numpy array): Array containing prediction scores.
    - ground_truths (numpy array): Array containing ground truth labels.
    - k (int): The number of top predictions to consider for calculating recall.

    Returns:
    - float: The recall at k score.
    """
    print(f"%%%"*20)
    print("You are now in calculate_recall_at_k function")
    print(f"Shape of predictions: {predictions.shape}")
    print(f"Shape of ground_truths: {ground_truths.shape}")
    print(f"%%%"*20)

    # Get the indices of the top k predictions
    top_k_indices = predictions.argsort()[:, -k:]
    
    # Initialize variable to store the cumulative recall
    cumulative_recall = 0.0
    
    # Loop over all queries
    for i in range(len(predictions)):
        # Get the relevant indices for the current query
        relevant_indices = np.where(ground_truths[i])[0]
        
        # Get the top k predictions for the current query
        top_k_preds = top_k_indices[i]
        
        # Calculate the recall for the current query
        print("##"*20)
        print(f"length of relevant_indices = {len(relevant_indices)}")
        print(f"length of top_k_preds = {len(top_k_preds)}")
        recall = len(set(top_k_preds) & set(relevant_indices)) / len(relevant_indices)
        cumulative_recall += recall
    
    # Calculate the mean recall at k
    recall_at_k = cumulative_recall / len(predictions)
    
    return recall_at_k
