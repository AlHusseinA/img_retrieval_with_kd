import numpy as np
import torch

def normalize_features(features):
    """
    Normalizes the features to unit length.
    
    Args:
        features (torch.Tensor): A 2D tensor of features.

    Returns:
        torch.Tensor: The normalized features.
    """
    return features / features.norm(dim=1, keepdim=True)



def average_precision(retrieved, relevant):
    """
    Calculate Average Precision for a single query.
    :param retrieved: List of retrieved item indices.
    :param relevant: List of relevant item indices.
    :return: Average Precision score.
    """
    retrieved = np.array(retrieved)
    relevant = np.array(relevant)
    rel_mask = np.in1d(retrieved, relevant)

    cum_rel = np.cumsum(rel_mask)
    precision_at_k = cum_rel / (np.arange(len(retrieved)) + 1)
    average_precision = np.sum(precision_at_k * rel_mask) / len(relevant)
    
    return average_precision

def mean_average_precision(retrieved_lists, relevant_lists):
    """
    Calculate Mean Average Precision (mAP) for a set of queries.
    :param retrieved_lists: List of lists, each containing retrieved item indices for a query.
    :param relevant_lists: List of lists, each containing relevant item indices for a query.
    :return: Mean Average Precision score.
    """
    ap_scores = [average_precision(retrieved, relevant)
                 for retrieved, relevant in zip(retrieved_lists, relevant_lists)]
    
    return np.mean(ap_scores)



def recall_at_k(retrieved, relevant, k):
    """
    Calculate Recall at K for a single query.
    :param retrieved: List of retrieved item indices.
    :param relevant: List of relevant item indices.
    :param k: The number of top results to consider.
    :return: Recall at K score.
    """
    retrieved_top_k = set(retrieved[:k])
    relevant_set = set(relevant)
    recall = len(retrieved_top_k.intersection(relevant_set)) / len(relevant_set)
    return recall

def mean_recall_at_k(retrieved_lists, relevant_lists, k):
    """
    Calculate Mean Recall at K for a set of queries.
    :param retrieved_lists: List of lists, each containing retrieved item indices for a query.
    :param relevant_lists: List of lists, each containing relevant item indices for a query.
    :param k: The number of top results to consider for each query.
    :return: Mean Recall at K score.
    """
    recall_scores = [recall_at_k(retrieved, relevant, k)
                     for retrieved, relevant in zip(retrieved_lists, relevant_lists)]
    
    return np.mean(recall_scores)




