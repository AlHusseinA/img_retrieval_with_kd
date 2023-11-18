import sys
sys.path.append('/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp/src')
print(sys.path)
import torch
import torch.nn as nn
import os
from torchvision.models import resnet50, ResNet50_Weights
from datetime import datetime
import yaml
from dataloaders.cub200loader import DataLoaderCUB200
# from utils.metrics import calculate_map, calculate_recall_at_k
import models.resnet50 as resnet50
from utils.debugging_functions import create_subset_data
from exp_logging.metricsLogger import MetricsLogger

# from utils.gallery import extract_gallery_features
# from utils.similarities import evaluate_on_retrieval
from tests.similarities_debug import evaluate_on_retrieval
from utils.features_unittest import TestFeatureSize
import numpy as np
import random

# src/tests/similarties_debug.py
# /home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp/src/tests/similarties_debug.py



def main_resnet():
    # Set seed for reproducibility
    seed_value = 42  # can be any integer value
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)
    
    def ensure_directory_exists(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory {directory_path} created.")
        else:
            print(f"Directory {directory_path} already exists.")




    
    #### set device #####
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#"*30)
    print(f"\nYou are curently using {device} device")
    print(f"\nYou are currenyly running retrieval ONLY mode!\n")
    print("#"*30)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    #### set the confi dict #####
    # Initialize config dictionary
    config_dict = {}
    #### set device #####
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')  # Default to '0' if not set
    print(f"GPU ID: {gpu_id}")
    #### directory to save fine tuned weights #####
    # save_dir = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned"
    # for models already moved to a different folder
    # load_dir ="/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/best_performing_04102023 evening"
    # load_dir = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiment_gpu_1/weights"
    load_dir = f"/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiment_gpu_{gpu_id}/weights"
    data_root ="/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/cub200/"
    # Create dictionaries to store metrics
    metrics_cub200_retrieval = {}

    ensure_directory_exists(load_dir)
           
    #### hyperparameters #####
    batch_size  = 128
    epochs = 1
    lr=0.0009
    use_early_stopping=True


    #### print hyperparameters #####
    print("/\\"*30)
    print(f"You are using batch size: {batch_size}")
    print(f"You are using epochs: {epochs}")
    print(f"You are using early stopping: {use_early_stopping}")
    print("/\\"*30)

    #### get data #####root, batch_size=32,num_workers=10   
    DEBUG_MODE = False


    #### get data #####root, batch_size=32,num_workers=10   
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    num_classes_cub200 = dataloadercub200.get_number_of_classes()

    if DEBUG_MODE:
        # create small subset of data to make debuggin faster
        trainloader_cub200_dump, testloader_cub200_dump = dataloadercub200.get_dataloaders()
        trainloader_cub200, testloader_cub200 = create_subset_data(trainloader_cub200_dump, testloader_cub200_dump, batch_size=32)
    else:
        trainloader_cub200, testloader_cub200 = dataloadercub200.get_dataloaders()


    # Initializing metrics logger for later use in logging metrics
    dataset_names=["cub200"]



    #### feature sizes #####
    # feature_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    feature_sizes = [2048]
    # Get current date and time

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    #### set the confi dict #####
    # Initialize config dictionary
    config_dict = {}
    config_dict["learning_rate"] = lr
    config_dict["batch_size"] = batch_size
    config_dict["epochs"] = epochs
    # config_dict["use_early_stopping"] = use_early_stopping
    config_dict["feature_sizes"] = feature_sizes
    config_dict["dataset_names"] = dataset_names
    config_dict["start_time"] = current_time 
    config_dict["seed_value"] = seed_value
    # Create the directory path in a platform-independent way
    directory_path_conf = os.path.join("src", "configs")
    # Check if the directory exists; if not, create it
    if not os.path.exists(directory_path_conf):
        os.makedirs(directory_path_conf)
    # Create a dynamic filename based on current time and learning rate
    filename = f"config__retrieval_only_script_{current_time}_lr_{lr}_batch_{batch_size}.yaml"
    full_path = os.path.join(directory_path_conf, filename)

    # Write to the dynamically named config.yaml file
    with open(full_path, "w") as f:
        yaml.dump(config_dict, f)


    #retrieval
    metrics_logger_retrieval = MetricsLogger() 

    # metrics_logger_retrieval.to(device)
    for feature_size in feature_sizes:

        model_cub200 = resnet50.ResNet50(feature_size, num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)
        fine_tuned_weights = torch.load(f'{load_dir}/resnet50_{feature_size}_CUB-200_batchsize_{batch_size}_lr_{lr}.pth')
                            # torch.save(f'{save_dir}/resnet50_{feature_size}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth')
        model_cub200.load_state_dict(fine_tuned_weights)
        model_cub200.feature_extractor_mode()

        #unit test for feature size
        testing_size = TestFeatureSize(model_cub200, feature_size) # this will confirm that the feature size is correct

        try:
            testing_size.test_feature_size()
            print(f"The model under evaluation is in indeed with {feature_size} feature size!")

        except AssertionError as e:
            # add an error message to the assertion error
            e.args += (f"Expected feature size {feature_size}, got {model_cub200.features_out.in_features}")   
            raise e # if the feature size is not correct, raise an error
        
        retrieval_metrics_cub200 = {}
        model_cub200 = model_cub200.to(device)
        retrieval_metrics_cub200 = evaluate_on_retrieval(model_cub200, trainloader_cub200, testloader_cub200, batch_size, device=device) 
        print(f"\nRetrieval metrics for CUB-200: {retrieval_metrics_cub200}")
        metrics_logger_retrieval.log_retrieval_metrics(feature_size, retrieval_metrics_cub200, dataset_names[0])
        # print(f"\nRetrieval metrics for CUB-200 against feature size: {retrieval_metrics_cub200}")
        print(f"Retrieval evaluation for CUB-200 with feature size {feature_size} is complete!\n")
        retrieval_metrics_cub200[feature_size] = {'mAP': retrieval_metrics_cub200['mAP'], 'R@1': retrieval_metrics_cub200['R@1'], 
                                    'R@5': retrieval_metrics_cub200['R@5'], 'R@10': retrieval_metrics_cub200['R@10']}


if __name__ == "__main__":
    main_resnet()



