

import numpy as np
import torch
from torch.nn import CosineSimilarity
from exp_logging.metricsLogger import MetricsLogger
from utils.helpers_functions_retrieval import normalize_features, compute_map
import time

from tqdm import tqdm
from torchmetrics.retrieval import RetrievalMAP, RetrievalRecall


def image_retrieval(single_query_feature, gallery_features, device=None):
    """
    This function takes a query feature and a set of gallery features and computes the similarity scores 
    between the query and each gallery feature using cosine similarity. It then sorts these scores 
    (and corresponding indices) in descending order.
    """
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
    


    # Normalize the query and gallery features to unit length
    single_query_feature = single_query_feature / single_query_feature.norm(dim=1, keepdim=True)
    gallery_features = gallery_features / gallery_features.norm(dim=1, keepdim=True)


    # Create an instance of the torch.nn.CosineSimilarity class and compute cosine similarity scores
    similarity_scores = []
    cos_sim = CosineSimilarity(dim=1, eps=1e-6)

    # Move tensors to the device
    single_query_feature = single_query_feature.to(device)
    gallery_features = gallery_features.to(device)   

    similarity_scores = cos_sim(gallery_features, single_query_feature)

    sorted_scores, sorted_indices = torch.sort(similarity_scores, descending=True)


    # Initialize rank matrix
    rank_matrix = torch.empty(len(gallery_features), 1, dtype=torch.long)

    # Fill in the rank matrix
    for rank, index in enumerate(sorted_indices):
        rank_matrix[index] = rank + 1  # Rank is 1-indexed

    return similarity_scores, sorted_scores, sorted_indices, rank_matrix



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

    model.eval()


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
                # ##########################################  
                similarity_scores, sorted_scores, sorted_indices, rank_matrix   = image_retrieval(single_query_feature, gallery_features, device)
                ##########################################             
                ground_truths = (gallery_labels == single_query_label).int()



                indexes_tensor = torch.full_like(similarity_scores2, query_idx_counter, dtype=torch.long)
                
                rmap = compute_map(rank_matrix, ground_truths)




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

        if batch_counter % N == 0:
            print(f"Processing batch {batch_counter} of {total_batches} in this image retrieval evaluation")

    
    metrics = {



        }

    return metrics


