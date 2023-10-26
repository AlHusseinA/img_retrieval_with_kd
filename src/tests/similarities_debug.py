import numpy as np
import torch
from torch.nn import CosineSimilarity
from exp_logging.metricsLogger import MetricsLogger
# from utils.metrics import calculate_map, calculate_recall_at_k
from sklearn.metrics import average_precision_score, recall_score

import time

from tqdm import tqdm
from torchmetrics.retrieval import RetrievalMAP, RetrievalRecall

def image_retrieval(single_query_feature, gallery_features, device=None):

    if device==None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device



    if len(single_query_feature.shape) == 1:
        single_query_feature = single_query_feature.unsqueeze(0)

    # Check if the feature dimensions of query and gallery are the same
    if len(single_query_feature.shape) < 2 or len(gallery_features.shape) < 2:
        raise ValueError("Both query and gallery features must be 2D tensors.")
    # Check if the feature dimensions of query and gallery are the same
    if single_query_feature.shape[1] != gallery_features.shape[1]:
        raise ValueError("The feature dimensions of the query and gallery must match.")


    # Ensure the query and gallery feature arrays are non-empty and have the appropriate dimensions
    if single_query_feature.size == 0 or gallery_features.size == 0:
        raise ValueError("Query feature and gallery features cannot be empty")
    
    if len(single_query_feature.shape) != 2 or len(gallery_features.shape) != 2:
        raise ValueError("Query feature and gallery features must be 2D arrays")
    
    # Create an instance of the torch.nn.CosineSimilarity class and compute cosine similarity scores
    similarity_scores = []
    cos_sim = CosineSimilarity(dim=1, eps=1e-6)

    # Move tensors to the device
    single_query_feature = single_query_feature.to(device)
    gallery_features = gallery_features.to(device)   
    
    
    similarity_scores = cos_sim(gallery_features, single_query_feature)

    sorted_scores, sorted_indices = torch.sort(similarity_scores, descending=True)
    
    return similarity_scores, sorted_scores, sorted_indices

def create_gallery_features(model, trainloader, batch_size=32, shuffle=False, device=None):

    model.eval()
    counter=0
    gallery_features, gallery_labels = [], []

    pbar = tqdm(total=len(trainloader), desc="Creating Gallery features", unit="Batches")

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        counter+=1

        with torch.inference_mode():
            output = model(images)

        gallery_features.append(output)
        gallery_labels.extend(labels.tolist())
        pbar.update(1)
        pbar.set_postfix({"Message": f"Creating Gallery features {counter}"})
        

    gallery_features = torch.vstack(gallery_features)
    gallery_labels = torch.tensor(gallery_labels, device=device)    

    return gallery_features, gallery_labels



# def evaluate_on_retrieval(model, trainloader, testloader, metrics_logger_retrieval, batch_size=32, shuffle=False, device=None):
def evaluate_on_retrieval(model, trainloader, testloader, batch_size=32, shuffle=False, device=None):

    """

    Evaluate a model on image retrieval task.
    """
    if device==None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device

    print(f"Device for retrieval evaluation: {device}")
    total_batches = len(testloader)  # Total number of batches in testloader
    print(f"Total number of batches in the testlaoder: {total_batches} of this image retrieval evaluation")   
    
    N = 20  # Number of batches after which to print the batch number   
    batch_counter = 0  # Initialize a counter for the batches


    # 2. Extract features for gallery
    gallery_features, gallery_labels = create_gallery_features(model, trainloader, batch_size=batch_size, shuffle=shuffle, device=device)

    # Initialize metrics
    # mAPs = []
    mAPs_sklearn = []  # For storing mAP values calculated using scikit-learn
    mAPs_custom = []  # For storing mAP values calculated using custom function
    Rs = {1: [], 5: [], 10: []}
    Rs_sklearn = {1: [], 5: [], 10: []}  # For storing Recall@k values calculated using scikit-learn

    # all_query_features = []
    model.eval()
    rmap = RetrievalMAP()
    r1 = RetrievalRecall(top_k=1)
    r5 = RetrievalRecall(top_k=5)
    r10 = RetrievalRecall(top_k=10)    # r2 = RetrievalRecall(k=20)   # top 20
    query_idx_counter = 0  # Initialize a counter to keep track of query indices across batches

    counter=0
    TEST_FLAG=0
    # start_time = time.time()
    pbar = tqdm(total=len(testloader), desc="Calculating mAP/Rs", unit="Batches")

    for query_images, query_labels in testloader:
        query_images, query_labels = query_images.to(device), query_labels.to(device)
        counter+=1
        if TEST_FLAG==0:
            start_time = time.time()
            # TEST_FLAG=1

        with torch.inference_mode():
            # features for a batch of query images passed through the model
            query_features = model(query_images)         
             # and in the batch you just producd, take each image features one by one as a query and test image retrieval
            for single_query_feature, single_query_label in zip(query_features, query_labels):
                # # because zip will turn single_query_lable to a scalar tensor
                # ##########################################  
                similarity_scores2, _, _ = image_retrieval(single_query_feature, gallery_features, device)   
                ##########################################             
                ground_truths = (gallery_labels == single_query_label).int()
                ##########################################
                mAP_sklearn = average_precision_score(ground_truths.cpu().numpy(), similarity_scores2.cpu().detach().numpy())
                mAPs_sklearn.append(mAP_sklearn)
                #########
                mAP_custom = mean_average_precision(ground_truths.cpu().numpy(), similarity_scores2.cpu().detach().numpy())
                mAPs_custom.append(mAP_custom)
                #########
                recall_values = calculate_recall_at_k(similarity_scores2.unsqueeze(0), ground_truths.unsqueeze(0))
                for k, val in recall_values.items():
                    Rs[k].append(val)
                #########
                # Calculate Recall@k using scikit-learn
                sorted_indices = np.argsort(similarity_scores2.cpu().detach().numpy())[::-1]
                for k in [1, 5, 10]:
                    top_k_indices = sorted_indices[:k]
                    top_k_ground_truths = ground_truths.cpu().numpy()[top_k_indices]
                    recall_k_sklearn = recall_score(top_k_ground_truths, np.ones_like(top_k_ground_truths), zero_division=1)
                    Rs_sklearn[k].append(recall_k_sklearn)
                ##################################


                indexes_tensor = torch.full_like(similarity_scores2, query_idx_counter, dtype=torch.long)

                
                # Store or compare these recall values
                

                rmap.update(preds=similarity_scores2, target=ground_truths, indexes=indexes_tensor)
                r1.update(preds=similarity_scores2, target=ground_truths, indexes=indexes_tensor)
                r5.update(preds=similarity_scores2, target=ground_truths, indexes=indexes_tensor)
                r10.update(preds=similarity_scores2, target=ground_truths, indexes=indexes_tensor)
                # Rs[1].append(r1(preds=similarity_scores2, target=ground_truths, indexes=indexes_tensor).item())
                # Rs[5].append(r5(preds=similarity_scores2, target=ground_truths, indexes=indexes_tensor).item())
                # Rs[10].append(r10(preds=similarity_scores2, target=ground_truths, indexes=indexes_tensor).item())

                query_idx_counter += 1  # Increment the query index counter


    
        pbar.update(1)
        pbar.set_postfix({"Message": f"Calculating retrieval metrics for batch {counter}"})
    
        if TEST_FLAG==0:
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_time_minutes = elapsed_time / 60  # <-- Convert to minutes
            elapsed_time_hours = elapsed_time / 3600  # <-- Convert to hours
            print(f"Time taken to process this batch of {batch_counter} images/features: {elapsed_time:.4f} seconds, {elapsed_time_minutes:.4f} minutes, {elapsed_time_hours:.4f} hours")
            # TEST_FLAG=1

            


        batch_counter += 1  # Increment the batch counter


        # if batch_countgiven 

    final_map = rmap.compute()

    final_r1 = r1.compute()
    final_r5 = r5.compute()
    final_r10 = r10.compute()

    # 5. Return the metrics as a dictionary
    # metrics = {
    #     'mAP': torch.mean(torch.tensor(final_map, device=device)).item(),
    #     'R@1': torch.mean(torch.tensor(Rs[1], device=device)).item(),
    #     'R@5': torch.mean(torch.tensor(Rs[5], device=device)).item(),
    #     'R@10': torch.mean(torch.tensor(Rs[10], device=device)).item(), }
    
    # metrics = {
    #     'mAP': torch.mean(torch.tensor(final_map, device=device)).item(),
    #     'R@1': torch.mean(torch.tensor(Rs[1], device=device)).item(),
    #     'R@5': torch.mean(torch.tensor(Rs[5], device=device)).item(),
    #     'R@10': torch.mean(torch.tensor(Rs[10], device=device)).item(), }
    

    

    # Compute the final average Recall@k values
    
    final_recall_at_1 = sum(Rs[1]) / len(Rs[1])
    final_recall_at_5 = sum(Rs[5]) / len(Rs[5])
    final_recall_at_10 = sum(Rs[10]) / len(Rs[10])
    # print(f"\n\nFinal Recall@1: {final_recall_at_1:.4f}")
    # print(f"Final Recall@5: {final_recall_at_5:.4f}")
    # print(f"Final Recall@10: {final_recall_at_10:.4f}")

    final_mAP_sklearn = np.mean(mAPs_sklearn)
    final_recall_at_1_sklearn = np.mean(Rs_sklearn[1])
    final_recall_at_5_sklearn = np.mean(Rs_sklearn[5])
    final_recall_at_10_sklearn = np.mean(Rs_sklearn[10])
    final_mAP_custom = np.mean(mAPs_custom)

    metrics = {
        'mAP': final_map.item(),
        'R@1': final_r1.item(),
        'R@5': final_r5.item(),
        'R@10': final_r10.item(),       
        }
    added_metrics =        { 'mAP_sklearn': final_mAP_sklearn,
        'mAP_custom': final_mAP_custom,
        'R1_costum_recall': final_recall_at_1,
        'R5_costum_recall': final_recall_at_5,
        'R10_costum_recall': final_recall_at_10,
        'R1_sklearn': final_recall_at_1_sklearn,
        'R5_sklearn': final_recall_at_5_sklearn,
        'R10_sklearn': final_recall_at_10_sklearn,}

    print(f"\n\n Added metrics: {added_metrics}")
    return metrics


# Mean Average Precision (mAP)
def mean_average_precision(y_true, y_score):
    sorted_indices = np.argsort(y_score)[::-1]
    sorted_y_true = y_true[sorted_indices]
    cumulative_sum = np.cumsum(sorted_y_true)
    cumulative_precision = cumulative_sum / (np.arange(len(y_true)) + 1)
    return np.sum(cumulative_precision * sorted_y_true) / np.sum(sorted_y_true)

# Recall@k
def recall_at_k(y_true, y_score, k):
    top_k_indices = np.argsort(y_score)[::-1][:k]
    return np.sum(y_true[top_k_indices]) / np.sum(y_true)



def calculate_recall_at_k(similarity_scores, ground_truths, k_values=[1, 5, 10]):
    """
    Calculate Recall@k for given similarity scores and ground truths.
    
    Args:
    - similarity_scores (Tensor): Similarity scores between queries and gallery. Shape [num_queries, num_gallery].
    - ground_truths (Tensor): Ground truth relevance labels for gallery w.r.t each query. Shape [num_queries, num_gallery].
    - k_values (List[int]): List of 'k' values for which to calculate Recall@k.
    
    Returns:
    - recall_dict (Dict[int, float]): Dictionary of Recall@k values.
    """
    
    recall_dict = {}
    num_queries = similarity_scores.shape[0]
    
    for k in k_values:
        correct_retrievals = 0
        total_relevant = 0
        
        for i in range(num_queries):
            # Sort gallery by similarity score for each query
            sorted_indices = torch.argsort(similarity_scores[i], descending=True)
            
            # Take top-k indices
            top_k_indices = sorted_indices[:k]
            
            # Count number of relevant items in top-k retrieved items
            correct_retrievals += ground_truths[i][top_k_indices].sum().item()
            
            # Count total number of relevant items for this query
            total_relevant += ground_truths[i].sum().item()
        
        # Calculate Recall@k for this k
        recall_at_k = correct_retrievals / total_relevant if total_relevant > 0 else 0
        recall_dict[k] = recall_at_k
    
    return recall_dict

