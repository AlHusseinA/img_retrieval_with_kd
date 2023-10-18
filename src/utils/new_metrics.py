import numpy as np
import torch
from torch.nn import CosineSimilarity
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import average_precision_score

def image_retrieval(single_query_feature, gallery_features, device):
    cos_sim = CosineSimilarity(dim=1, eps=1e-6)
    similarity_scores = cos_sim(gallery_features, single_query_feature)
    sorted_scores, sorted_indices = torch.sort(similarity_scores, descending=True)
    return sorted_scores, sorted_indices

def evaluate_on_retrieval(model, trainloader, testloader, device):
    model.feature_extractor_mode()
    gallery_features, gallery_labels = [], []
    model.eval()
    
    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        with torch.inference_mode():
            output = model(images)
        gallery_features.append(output)
        gallery_labels.extend(labels.tolist())

    gallery_features = torch.vstack(gallery_features)
    gallery_labels = torch.tensor(gallery_labels, device=device)

    APs = defaultdict(list)
    Rs = {1: [], 5: [], 10: []}

    for query_images, query_labels in testloader:
        query_images, query_labels = query_images.to(device), query_labels.to(device)
        with torch.inference_mode():
            query_features = model(query_images)

        for single_query_feature, single_query_label in zip(query_features, query_labels):
            sorted_scores, sorted_indices = image_retrieval(single_query_feature, gallery_features, device)
            ground_truths = (gallery_labels == single_query_label).int()

            # Calculate AP
            ap = average_precision_score(ground_truths.cpu().numpy(), sorted_scores.cpu().detach().numpy())
            APs[1].append(ap)  # Here, 1 is just a placeholder key. You can make it more meaningful if you wish.

            # Calculate Recall@k
            for k in [1, 5, 10]:
                Rk = len(set(sorted_indices[:k].cpu().numpy()) & set(np.where(ground_truths.cpu().numpy())[0])) / max(len(np.where(ground_truths.cpu().numpy())[0]), 1)
                Rs[k].append(Rk)

    metrics = {
        'mAP': np.mean(np.array(APs[1])),
        'R@1': np.mean(np.array(Rs[1])),
        'R@5': np.mean(np.array(Rs[5])),
        'R@10': np.mean(np.array(Rs[10]))
    }

    return metrics

# The average_precision_score function from scikit-learn can replace the calculate_map function.
