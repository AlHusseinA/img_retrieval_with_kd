
import numpy as np
import torch
from torch.nn import CosineSimilarity
# from exp_logging.metricsLogger import MetricsLogger
# from utils.helpers_functions_retrieval import normalize_features
import time
from tqdm import tqdm
from torchmetrics.retrieval import RetrievalMAP, RetrievalRecall
from test_debug_retrieval_prob.visualize_retrieval import visualize_retrieval_first_n_samples

def image_retrieval(single_query_feature, gallery_features, gallery_labels, query_label, debug_flag, device=None):
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
    

#############################################################################################################
#############################################################################################################

    # single_query_feature = single_query_feature.unsqueeze(0) if len(single_query_feature.shape) == 1 else single_query_feature
    

    # similarity_scores = []
 
    # cos_sim = CosineSimilarity(dim=1, eps=1e-6)
    # all_similarities = cos_sim(gallery_features, single_query_feature)

    # # Find the index of the maximum similarity (assumed to be the query image itself)
    # max_similarity_index = torch.argmax(all_similarities)

    # # Create a mask to exclude the query image
    # mask = torch.ones(all_similarities.shape, dtype=torch.bool, device=device)
    # mask[max_similarity_index] = False

    # # Apply the mask to exclude the query image from similarity scores
    # masked_similarities = all_similarities[mask]

    # # Sort the masked similarities
    # # ex: sorted_indices[:15]=tensor([  30,   36,   56,   52,   31,   39,   18,   49,   73,  572,   53,   51, 42,   24, 2012], device='cuda:0')
    # # sorted_scores[:15]=tensor([0.8610, 0.8580, 0.8434, 0.8383, 0.8304, 0.8255, 0.8165, 0.8161, 0.8127, 0.8060, 0.8059, 0.8059, 0.8052, 0.8041, 0.8036], device='cuda:0')
    # sorted_scores, sorted_indices = torch.sort(masked_similarities, descending=True)
    
    # # print(f"####"*25)
    # # print(f"{sorted_scores[:15]=}")
    # # print(f"{sorted_indices[:15]=}")
    # # exit(f"###"*25)

    # # Get the valid indices after masking
    # # This will return the indices of the gallery images that are not the query image
    # # So its size will be (gallery_features.size(0) - 1)
    # # ex: valid_indices[:15]=tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15], device='cuda:0')
    # valid_indices = torch.arange(gallery_features.size(0), device=device)[mask]

    # assert valid_indices.size(0) == gallery_features.size(0) - 1, "Valid indices size must be equal to gallery_features.size(0) - 1"
    # assert valid_indices.size(0) == masked_similarities.size(0), "Valid indices size must be equal to masked_similarities.size(0)"
    # assert valid_indices.size(0) == sorted_scores.size(0), "Valid indices size must be equal to sorted_scores.size(0)"
    # assert valid_indices.size(0) == sorted_indices.size(0), "Valid indices size must be equal to sorted_indices.size(0)"
    # print(f"{valid_indices[:15]=}")
    # print(f"{sorted_indices[:15]=}")
    # exit(f"{sorted_scores[:15]=}")
#############################################################################################################
#############################################################################################################
    # Assuming single_query_feature and gallery_features are available
    # single_query_feature: Feature vector of the query image
    # gallery_features: Tensor containing feature vectors of all gallery images

    # Ensure single_query_feature is a 2D tensor for batch operation
    single_query_feature = single_query_feature.unsqueeze(0) if len(single_query_feature.shape) == 1 else single_query_feature

    # Calculate cosine similarities
    cos_sim = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    all_similarities = cos_sim(gallery_features, single_query_feature)

    # Direct comparison to find the exact match
    exact_match_index = None
    for idx, feature in enumerate(gallery_features):
        if torch.equal(feature, single_query_feature.squeeze(0)):
            exact_match_index = idx
            break

    # Validate if an exact match was found
    if exact_match_index is None:
        raise ValueError("No exact match found in the gallery for the query image")

    # Create a mask to exclude the query image
    mask = torch.ones(all_similarities.shape, dtype=torch.bool, device=gallery_features.device)
    mask[exact_match_index] = False

    # Proceed with masked similarities
    masked_similarities = all_similarities[mask]
    sorted_scores, sorted_indices = torch.sort(masked_similarities, descending=True)

    # Correct way to get valid indices
    valid_indices = torch.arange(gallery_features.size(0), device=gallery_features.device)[mask]

    # Assertions
    assert valid_indices.size(0) == gallery_features.size(0) - 1, "Valid indices size must be equal to gallery_features.size(0) - 1"
    # Further assertions as in your original code
    print(f"masked_similarities[:15]= {masked_similarities[:15]}")
    print(f"{valid_indices[:15]=}")
    print(f"{sorted_indices[:15]=}")
    exit(f"{sorted_scores[:15]=}")

    return masked_similarities, sorted_scores, sorted_indices, valid_indices



def create_gallery_features(model, dataloader, batch_size, shuffle=False, device=None):

    model.eval()
    counter=0
    gallery_features, gallery_labels = [], []


    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        counter+=1

        with torch.inference_mode():
            output = model(images)

        gallery_features.append(output)
        gallery_labels.extend(labels.tolist())


    gallery_features = torch.vstack(gallery_features)
    gallery_labels = torch.tensor(gallery_labels, device=device)    

    return gallery_features, gallery_labels



def evaluate_on_retrieval_test_show_res(model, testloader, filtered_loader, label_to_name_test, batch_size, shuffle=False, device=None):

    if device==None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = device

    print(f"Device for retrieval evaluation: {device}")
    total_batches = len(testloader)  # Total number of batches in testloader
    print(f"Total number of batches in the testlaoder: {total_batches} of this image retrieval evaluation")   

    print(f"####"*25)
    # total_images_train = len(trainloader_cub200.dataset)
    total_images_test = len(testloader.dataset)
    # print(f"Total images in trainloader: {total_images_train}")
    print(f"Total images in testloader: {total_images_test}")
    print(f"####"*25)
    N = 20  # Number of batches after which to print the batch number   


    gallery_features, gallery_labels = create_gallery_features(model, testloader, batch_size=batch_size, shuffle=shuffle, device=device)


    model.eval()
    # As per standard,we measure the model performance with mean Average Precision(mAP)at top1000(mAP@1K) for single-labeled datasets (i.e.CUB200, ImageNet1000, andCIFAR10)
    rmap = RetrievalMAP(top_k=100)
    r1 = RetrievalRecall(top_k=1)
    r5 = RetrievalRecall(top_k=5)
    r10 = RetrievalRecall(top_k=10)    # r2 = RetrievalRecall(k=20)   # top 20
    query_idx_counter = 0  # Initialize a counter to keep track of query indices across batches

    DEBUG_dir = "/home/alabutaleb/Desktop/confirmation/Retrieval_eval_baselines_experiment_gpu_0"
    counter=0
    TEST_FLAG=0
    # start_time = time.time()

    
    processed_classes = set()
    class_queries = {}  # To store the first query index of each class

    # for query_idx, (query_images, query_labels) in enumerate(filtered_loader):
    #     query_images, query_labels = query_images.to(device), query_labels.to(device)
    #     counter+=1

    #     for idx, label in enumerate(query_labels):
    #         if label.item() not in processed_classes:
    #             processed_classes.add(label.item())
    #             class_queries[label.item()] = query_idx * batch_size + idx

    #         if len(processed_classes) >= 10:

    #             break

    #     if len(processed_classes) >= 10:
    #         break

        # with torch.inference_mode():
        #     # features for a batch of query images passed through the model
        #     query_features = model(query_images)  

        #     for single_query_feature, single_query_label in zip(query_features, query_labels):
        #         # Exclude the query itself during retrieval
        #         similarity_scores2, _, _, valid_indices = image_retrieval(single_query_feature, gallery_features, gallery_labels, single_query_label, counter, device)   
        #         # ground_truths = (gallery_labels == single_query_label.unsqueeze(0)).int()
        #         ground_truths = (gallery_labels == single_query_label).int()

        #         # Filter ground truths and indexes_tensor to match valid_indices
        #         valid_ground_truths = ground_truths[valid_indices]
        #         valid_indexes_tensor = torch.arange(gallery_features.size(0), device=device)[valid_indices]

                            
        #         # print(f"###"*25)
        #         # print(f"{similarity_scores2.shape=}")
        #         # print(f"{valid_indices.shape=}")
        #         # print(f"{ground_truths.shape=}")
        #         # print(f"{valid_ground_truths.shape=}")
        #         # print(f"{valid_indexes_tensor.shape=}")
        #         # print(f"type(similarity_scores2): {type(similarity_scores2)}")
        #         # print(f"type(ground_truths): {type(ground_truths)}")
        #         # print(f"type(valid_ground_truths): {type(valid_ground_truths)}")
        #         # exit("exit!!!")

        #         indexes_tensor = torch.full_like(similarity_scores2, query_idx_counter, dtype=torch.long)
        #         rmap.update(preds=similarity_scores2, target=valid_ground_truths, indexes=valid_indexes_tensor)
        #         r1.update(preds=similarity_scores2, target=valid_ground_truths, indexes=valid_indexes_tensor)
        #         r5.update(preds=similarity_scores2, target=valid_ground_truths, indexes=valid_indexes_tensor)
        #         r10.update(preds=similarity_scores2, target=valid_ground_truths, indexes=valid_indexes_tensor)

        #         query_idx_counter += 1  # Increment the query index counter

      
            
    retrieved_indices ={}

    for query_idx, (query_images, query_labels) in enumerate(filtered_loader):
        # print(f"###"*25)
        # print(f"{query_labels= }")
        # print(f"{query_idx=}")
        # exit(f"{query_images.shape=}")

        # batch_counter += 1  # Increment the batch counter
        for idx, label in enumerate(query_labels):
            if label.item() not in processed_classes:
                processed_classes.add(label.item())
                class_queries[label.item()] = query_idx * batch_size + idx
                print(f"XXXX"*25)
                print(f"query_idx: {query_idx}")
                print(f"idx: {idx}")
                print(f"batch_size: {batch_size}")
                print(f"{query_idx * batch_size + idx=}")
                print(f"{class_queries[label.item()]=}")
                print(f"XXXX"*25)

            if len(processed_classes) >= 10:

                break

        if len(processed_classes) >= 10:
            break
        print(f"###"*25)
        print(f"type of class_queries: {type(class_queries)}")
        print(f"Size of processed_classes: {len(processed_classes)}")
        print(f"{processed_classes=}")
        print(f"{type(class_queries)=}")
        print(f"{class_queries=}")
        print(f"{class_queries.items()=}")

    with torch.inference_mode():

        for class_label, query_idx in class_queries.items():
            # print(f"###"*25)
            # print(f"{query_idx= }")
            # print(f"{class_queries.items()=}")
            # exit(f"{class_label=}")

            # Load the specific query image and label
        # for query_idx, (query_image, query_label) in enumerate(filtered_loader):

            query_image, query_label = filtered_loader.dataset[query_idx]
            # query_image, query_label = query_image.to(device).unsqueeze(0), torch.tensor([query_label]).to(device)
            query_image, query_label = query_image.to(device).unsqueeze(0), torch.tensor([query_label]).to(device)

            # Perform model inference
            with torch.inference_mode():
                query_feature = model(query_image)

            # Image retrieval
            similarity_scores, _, _, valid_indices = image_retrieval(query_feature, gallery_features, gallery_labels, query_label, counter, device)
            valid_indexes_tensor = torch.arange(gallery_features.size(0), device=device)[valid_indices]
            top_retrieval_idxs = valid_indexes_tensor[:10]  # Top 10 retrieval results
            retrieved_indices[query_label.item()] = top_retrieval_idxs.cpu().numpy()
            # Visualization
            visualize_retrieval_first_n_samples(query_idx, top_retrieval_idxs.cpu().numpy(), testloader, DEBUG_dir, label_to_name_test)            

    for label, indices in retrieved_indices.items():
        # Convert the numpy array to a list for easy formatting
        indices_list = indices.tolist()
        # Create a string representation of the indices list
        indices_str = ', '.join(map(str, indices_list))
        # Print the class label and its corresponding indices
        print(f"Class Label: {label}, Top Retrieval Indices: [{indices_str}]")


    # just in case we need it        
    # all_query_features = torch.vstack(all_query_features) 
    # query_features = torch.cat((query_features, query_features), dim=0)  # Concatenate along the first dimension
    
    # final_map = rmap.compute()
    # final_r1 = r1.compute()
    # final_r5 = r5.compute()
    # final_r10 = r10.compute()

    # final_map = 0
    # final_r1 = 0
    # final_r5 = 0
    # final_r10 = 0

    # # 5. Return the metrics as a dictionary
    # metrics = {
 
    #     'mAP': final_map.item(),
    #     'R@1': final_r1.item(),
    #     'R@5': final_r5.item(),
    #     'R@10': final_r10.item(),

    #     }
    
    metrics = {
        'mAP': 0,
        'R@1': 0,
        'R@5': 0,
        'R@10': 0,
        }    

    return metrics

