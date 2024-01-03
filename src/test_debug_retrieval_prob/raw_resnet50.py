import sys
ROOT="/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp"
sys.path.append(f"{ROOT}/src")
print(sys.path)
import torch
import torch.nn as nn
import os

from exp_logging.metricsLogger import MetricsLogger
from torchvision.models import resnet50, ResNet50_Weights
from datetime import datetime
import yaml
from dataloaders.cub200loader import DataLoaderCUB200

import models.resnet50 as resnet50
from models.resnet50_vanilla import ResNet50_vanilla

from optimizers.sgd import SGDOptimizer, SGDOptimizerVariableLR
from optimizers.adam import AdamOptimizer
# from optimizers.adam_lr_var import AdamOptimizerVar
from optimizers.adam_lr_var_test import AdamOptimizerVar

from schedulers.cosine import CosineAnnealingLRWrapper, CosineAnnealingLRWrapperWithWarmup
from trainers.resnets_trainer import ResnetTrainer
from loss.ce import CustomCrossEntropyLoss
import matplotlib.pyplot as plt
# from utils.gallery import extract_gallery_features
from utils.similarities import evaluate_on_retrieval
from utils.similarities_no_torchmetrics import evaluate_on_retrieval_no_torchmetrics
# from utils.helper_functions import plot_multiple_metrics
from utils.features_unittest import TestFeatureSize
from utils.debugging_functions import create_subset_data, create_subset_data2
import numpy as np
import random

# for debugging
from torch.utils.data import Subset


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
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')  # Default to '0' if not set
    print(f"GPU ID: {gpu_id}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    #####################
    #### directory to save fine tuned weights #####
    base_directory = "/home/alabutaleb/Desktop/confirmation/"
    data_root ="/media/alabutaleb/data/cub200/"


    # Construct the dynamic directory path
    load_dir = f"/home/alabutaleb/Desktop/confirmation/baselines_allsizes/weights"
    #####################
    save_dir_folder = os.path.join(base_directory, f"Retrieval_eval_baselines_experiment_gpu_{gpu_id}")
    os.makedirs(save_dir_folder, exist_ok=True)
    save_dir = os.path.join(save_dir_folder, "weights")
    log_save_path = os.path.join(save_dir_folder, "logs_new")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_save_path, exist_ok=True)
    print(f"Weights will be loaded from: {load_dir}")
    print(f"Weight files will be saved in: {save_dir}")
    print(f"Log files will be saved in: {log_save_path}")
    #####################
   
    #####################
    print("#"*30)
    print(f"You are curently using {device} device")
    print("#"*30)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
 
    # Create dictionaries to store metrics
    metrics_cub200_finetuning = {}
    metrics_cub200_retrieval = {}

    ensure_directory_exists(save_dir_folder)
    ensure_directory_exists(log_save_path)
    ensure_directory_exists(save_dir)
           
    #### hyperparameters #####
    batch_size  = 256
    
    # epochs = 750
    # lr=0.00001      # lr=0.5
    # lr=0.000009
    # lr = 0.00007
    lr = 0.00007 # 
    # lr=0.01
    # lr = 0.000008

    weight_decay=2e-05

    # DEBUG_MODE = True
    DEBUG_MODE = False


    # use_early_stopping=True
    warmup_epochs=20 
    lr_warmup_decay=0.01 

    


    

    #### get data #####root, batch_size=32,num_workers=10   
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    num_classes_cub200 = dataloadercub200.get_number_of_classes()

    if DEBUG_MODE:
        # create small subset of data to make debuggin faster
        trainloader_cub200_dump, testloader_cub200_dump = dataloadercub200.get_dataloaders()
        trainloader_cub200, testloader_cub200, batch_size = create_subset_data2(trainloader_cub200_dump, testloader_cub200_dump, subset_ratio=0.1, batch_size=32)
        epochs = 5
        T_max=int(epochs/2)
        last_epoch = -1
    else:
        trainloader_cub200, testloader_cub200 = dataloadercub200.get_dataloaders()
        epochs = 1000
        T_max=int(epochs/2)
        last_epoch = -1
    
    #### print hyperparameters #####
    print("/\\"*30)
    print(f"Learning rate: {lr}")
    print(f"You are in DEBUG MODE: {DEBUG_MODE}")
    print(f"You are using batch size: {batch_size}")
    print(f"You are using epochs: {epochs}")
    print("/\\"*30)
    #### prep logger ####
    
    # Initializing metrics logger for later use in logging metrics
    dataset_names="cub200"

    metrics_logger = MetricsLogger()
    # metrics_logger.to(device)

    #### get loss for fine tuning #####
    criterion = CustomCrossEntropyLoss()

    #### feature sizes #####
    # feature_sizes = [2048]
    # feature_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    feature_sizes = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]




    # Get current date and time

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    


    # Plot the training and validation losses and accuracies in one plot each for all experiments
    
    #retrieval
    metrics_logger_retrieval = MetricsLogger() 
    retrieval_metrics_cub200 = {}


    # for feature_size in feature_sizes:
    print(f"##"*30)
    print(f"This code will test retrieval performance on a pretrained resnet50 vanilla with imagenet weights no finetuning done whatsoever!")
    print(f"##"*30)

    model_cub200 = ResNet50_vanilla(num_classes_cub200, set_eval_mode=False, weights=ResNet50_Weights.DEFAULT)
    # model_cub200 = ResNet50_convV2(feature_size, num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)
    # print(f"TEST: weights will be loaded from {load_dir}/resnet50_feature_size_{feature_size}_{dataset_names}_batchsize_{batch_size}_lr_{lr}")

    if DEBUG_MODE:
        batch_size = 256
    
    # fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_{feature_size}_{dataset_names}_batchsize_{batch_size}_lr_{lr}.pth')
    # save_dir = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiment_gpu_1/weights"
    # model_cub200.load_state_dict(fine_tuned_weights)
    model_cub200.feature_extractor_mode()

    #unit test for feature size
    # testing_size = TestFeatureSize(model_cub200, feature_size) # this will confirm that the feature size is correct

    # try:
    #     testing_size.test_feature_size()
    #     print(f"The model under evaluation is in indeed with {feature_size} feature size!")

    # except AssertionError as e:
    #     # add an error message to the assertion error
    #     e.args += (f"Expected feature size {feature_size}, got {model_cub200.features_out.in_features}")   
    #     raise e # if the feature size is not correct, raise an error
    
    

    model_cub200 = model_cub200.to(device)

    results = evaluate_on_retrieval(model_cub200, trainloader_cub200, testloader_cub200, batch_size, device=device)

    
    print(f"\nRetrieval metrics for CUB-200: {results}")
    metrics_logger_retrieval.log_retrieval_metrics(2048, results, dataset_names[0])
    # # print(f"\nRetrieval metrics for CUB-200 against feature size: {retrieval_metrics_cub200}")

    print(f"Retrieval evaluation for CUB-200 with feature size {2048} is complete!\n")
    retrieval_metrics_cub200[2048] = {'mAP': results['mAP'], 'R@1': results['R@1'], 
                                'R@5': results['R@5'], 'R@10': results['R@10']}
    

if __name__ == "__main__":
    main_resnet()

 
    

