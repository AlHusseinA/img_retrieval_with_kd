import numpy as np
import torch
from torch.nn import CosineSimilarity
from exp_logging.metricsLogger import MetricsLogger
from utils.metrics import calculate_map, calculate_recall_at_k
import time

from tqdm import tqdm
from torchmetrics.retrieval import RetrievalMAP, RetrievalRecall
# from torchmetrics.retrieval.base import RetrievalMetric

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
    
    # Create an instance of the torch.nn.CosineSimilarity class and compute cosine similarity scores
    similarity_scores = []
    cos_sim = CosineSimilarity(dim=1, eps=1e-6)

    # Move tensors to the device
    single_query_feature = single_query_feature.to(device)
    gallery_features = gallery_features.to(device)   

    similarity_scores = cos_sim(gallery_features, single_query_feature)
    # print(f"Shape of similarity_scores: {similarity_scores.shape}")
    # print(f"Type of similarity_scores: {type(similarity_scores)}")
    # print(f"Similarity scores: {similarity_scores[:10]}")
    # Get the sorted indices and sorted scores in descending order
    # I commented the line below on 5th Oct 7 pm to attempt to make the torchmetrics objectrs work
    sorted_scores, sorted_indices = torch.sort(similarity_scores, descending=True)

    # print("%%"*20)
    # print(f"Shape of sorted_scores: {sorted_scores.shape}")
    # print(f"First 10 scores: {sorted_scores[:10]}")
    # print(f"First 10 indices: {sorted_indices[:10]}")
    # print(f"Shape of sorted_indices: {sorted_indices.shape}")
    # print(f"Shape of single_query_feature: {single_query_feature.shape}")
    # print(f"Shape of gallery_features: {gallery_features.shape}")
    # print("%%"*20)
   

    # return sorted_scores, sorted_indices
    return similarity_scores, sorted_scores, sorted_indices

def create_gallery_features(model, trainloader, batch_size=32, shuffle=False, device=None):
    # print(f"XX"*20)
    # print(f"You are now in create_gallery_features function")
    # print(f"Device for creating gallery features: {device}")
    # print(f"XX"*20)
    # 2. Extract features for gallery
    model.eval()
    counter=0
    gallery_features, gallery_labels = [], []

    pbar = tqdm(total=len(trainloader), desc="Creating Gallery features", unit="Batches")

    for images, labels in trainloader:
    # for images, labels in tqdm(trainloader , desc="Creating gallery features", unit="Batches"):
        images, labels = images.to(device), labels.to(device)
        counter+=1
        # if flag_g==0:
        #     flag_g=1
        #     print(f"Size of this training batch in the trainloader is: {len(images)}")
        #     print(f"Size of the Trainloader is: {len(trainloader)}")
        #     print(f"Type of images: {type(images)}")
        #     print(f"Type of labels: {type(labels)}")
        #     print(f"Shape of images: {images.shape}")
        #     print(f"Shape of labels: {labels.shape}")
        #     print(f"size of labels: {len(labels)}")

        #     print(f"XX"*20)
        with torch.inference_mode():
            output = model(images)

        gallery_features.append(output)
        gallery_labels.extend(labels.tolist())
        pbar.update(1)
        pbar.set_postfix({"Message": f"Creating Gallery features {counter}"})
        
    # print(f"Size of gallery_features BEFORE VSTACK: {len(gallery_features)}")
    # print(f"Size of gallery_labels BEFORE VSTACK: {len(gallery_labels)}")
    # print(f"Type of gallery_features BEFORE VSTACK: {type(gallery_features)}")
    # print(f"Type of gallery_labels BEFORE VSTACK: {type(gallery_labels)}")
    # print(f"Shape of gallery_features BEFORE VSTACK: {gallery_features[0].shape=}")
    # print(f"Shape of gallery_features BEFORE VSTACK: {gallery_features[1].shape=}")
    gallery_features = torch.vstack(gallery_features)
    gallery_labels = torch.tensor(gallery_labels, device=device)    
    # # print(f"Size of gallery_features AFTER VSTACK: {len(gallery_features)}")
    # # print(f"Size of gallery_labels AFTER VSTACK: {len(gallery_labels)}")
    # print(f"Type of gallery_features AFTER VSTACK: {type(gallery_features)}")
    # print(f"Type of gallery_labels AFTER VSTACK: {type(gallery_labels)}")
    # print(f"Shape of gallery_features AFTER VSTACK: {gallery_features.shape}")
    # print(f"Shape of gallery_labels AFTER VSTACK: {gallery_labels.shape}")
    # print(f"XX"*20)
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
    mAPs = []
    # Rs = []
    Rs = {1: [], 5: [], 10: []}
    # all_query_features = []
    model.eval()
    rmap = RetrievalMAP()
    # r2 = RetrievalRecall(k=20)   # top 20
    counter=0
    TEST_FLAG=0
    # start_time = time.time()
    pbar = tqdm(total=len(testloader), desc="Calculating mAP/Rs", unit="Batches")

    for query_images, query_labels in testloader:
    # for query_images, query_labels in tqdm(testloader, desc="Calculating mAP/Rs", unit="Batches"):
    # for i, (query_images, query_labels) in enumerate(testloader):
        # if i != len(testloader) - 1:
        #     continue

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
                # sorted_scores, sorted_indices = image_retrieval(single_query_feature, gallery_features, device)   
                # ##########################################  
                similarity_scores2, _, _ = image_retrieval(single_query_feature, gallery_features, device)   
                ##########################################             
                # sorted_scores, sorted_indices = torch.sort(similarity_scores2, descending=True)
                ground_truths = (gallery_labels == single_query_label).int()
                # assert sorted_scores.dim() == 1, f"Expected sorted_scores to be 1D, got {sorted_scores.dim()}D"
                # assert ground_truths.dim() == 1, f"Expected ground_truths to be 1D, got {ground_truths.dim()}D"
                # assert sorted_scores.shape[0] == ground_truths.shape[0], "Shape mismatch between sorted_scores and ground_truths"
         

                non_zero_mask = similarity_scores2 != 0
                filtered_ground_truths = ground_truths[non_zero_mask]
                filtered_similarity_scores2 = similarity_scores2[non_zero_mask]
                non_zero_indices = torch.nonzero(filtered_similarity_scores2).squeeze()

                # mAP = rmap(preds=sorted_scores, target=ground_truths, indexes=sorted_indices)
                
                # non_zero_indices = torch.nonzero(similarity_scores2).squeeze()
                # ground_truths = ground_truths.type('torch.BoolTensor').to(device)
                # Create a tensor of integers from 0 to len(non_zero_mask) - 1
                # integer_indices = torch.arange(len(non_zero_mask))
                integer_indices = torch.arange(len(non_zero_mask), device=non_zero_mask.device)

                # Get the indices where the mask is True
                non_zero_indices_long = integer_indices[non_zero_mask].long()
                # mAP = rmap(preds=similarity_scores2, target=filtered_ground_truths, indexes=non_zero_indices)
                # mAP = rmap(preds=similarity_scores2, target=filtered_ground_truths, indexes=non_zero_mask)
                mAP = rmap(preds=filtered_similarity_scores2, target=filtered_ground_truths, indexes=non_zero_indices_long)

                mAPs.append(mAP)

                for k in [1, 5, 10]:
                    # Rk = calculate_recall_at_k(sorted_scores, ground_truths, k)
                    r2 = RetrievalRecall(top_k=k)   # top k matches
                    # Rk = r2(preds=sorted_scores, target=ground_truths, indexes=sorted_indices)
                    # if i != len(testloader) - 1:
                    #     continue
                    # print(f"XXX"*20)
                    # print(f"Shape of similarity_scores2: {similarity_scores2.shape}")
                    # print(f"Shape of filtered_similarity_scores2: {filtered_similarity_scores2.shape}")
                    # print(f"Shape of filtered_ground_truths: {filtered_ground_truths.shape}")
                    # print(f"Shape of non_zero_indices: {non_zero_indices.shape}")
                    # print(f"XXX"*20)
                    # Rk = r2(preds=similarity_scores2, target=filtered_ground_truths, indexes=non_zero_indices)
                    Rk = r2(preds=similarity_scores2, target=filtered_ground_truths, indexes=non_zero_indices_long)

                    # Rs[k].append(Rk)
                    Rs[k].append(Rk.item())  # Assuming Rk is a 0-dim tensor

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

        # if batch_counter >= N+1:
        #     print(f"Processed {batch_counter} batches out of {total_batches} in this image retrieval evaluation")
        #     break

        if batch_counter % N == 0:
            print(f"Processing batch {batch_counter} of {total_batches} in this image retrieval evaluation")

    # just in case we need it        
    # all_query_features = torch.vstack(all_query_features) 
    # query_features = torch.cat((query_features, query_features), dim=0)  # Concatenate along the first dimension

    # 5. Return the metrics as a dictionary
    metrics = {
        'mAP': torch.mean(torch.tensor(mAPs, device=device)).item(),
        'R@1': torch.mean(torch.tensor(Rs[1], device=device)).item(),
        'R@5': torch.mean(torch.tensor(Rs[5], device=device)).item(),
        'R@10': torch.mean(torch.tensor(Rs[10], device=device)).item(), }

    return metrics

