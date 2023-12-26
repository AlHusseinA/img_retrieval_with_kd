
import sys
import os

ROOT="/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp"
sys.path.append(f"{ROOT}/src")
print(f"{sys.path=}")
print(f"{os.getcwd()=}")
import torch
import torch.nn as nn
import torch.nn.init as init

from schedulers.cosine import CosineAnnealingLRWrapper
from optimizers.adam_for_fc import AdamOptimizerFC
from exp_logging.metricsLogger import MetricsLogger
from torchvision.models import resnet50, ResNet50_Weights
from datetime import datetime
import yaml
from dataloaders.cub200loader import DataLoaderCUB200
from compression.pca import PCAWrapper
# import models.resnet50 as resnet50
# from models.resnet50_conv_compession import ResNet50_conv
# from models.resnet50_conv_compressionV2 import ResNet50_convV2, ResNet50_convV3_BN
# from models.resnet50_vanilla_pca import ResNet50_vanilla_with_PCA
from models.resnet50_vanilla import ResNet50_vanilla

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# from utils.gallery import extract_gallery_features
from utils.similarities import evaluate_on_retrieval
# from utils.helper_functions import plot_multiple_metrics
from utils.features_unittest import TestFeatureSize
from utils.debugging_functions import create_subset_data
from utils.pca_fit_features import pca_fit_features
from utils.helpers_for_pca_exp import generate_and_process_features, make_predictions_model

# from trainers.train_eval_fc_pca import train_eval_fc_pca
from trainers.fc_trainer import fc_trainer
import numpy as np
import random


# for debugging
from torch.utils.data import Subset


def create_optimizer_fc(model, lr, weight_decay):        
    optimizer = AdamOptimizerFC(model, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    return optimizer.get_optimizer()

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")



# def load_resnet50_convV2(num_classes_cub200,feature_size, dataset_name, batch_size, lr, load_dir):
#     model = ResNet50_convV2(feature_size, num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)        
#     # file name = resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e-05.pth
#     fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e-05.pth')
#     model.load_state_dict(fine_tuned_weights)
#     model.feature_extractor_mode()
#     #unit test for feature size
#     testing_size = TestFeatureSize(model, feature_size) # this will confirm that the feature size is correct
#     try:
#         testing_size.test_feature_size()
#         print(f"The loaded model under evaluation is in indeed with {feature_size} feature size!")

#     except AssertionError as e:
#         # add an error message to the assertion error
#         e.args += (f"Expected feature size {feature_size}, got {model.features_out.in_features}")   
#         raise e # if the feature size is not correct, raise an error
#     return model

def load_resnet50_unmodifiedVanilla(num_classes_cub200,feature_size, dataset_name, batch_size, lr, load_dir):
    model = ResNet50_vanilla(num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)        
    # fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_{feature_size}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth')
    fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e.pth')

    # resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e

    model.load_state_dict(fine_tuned_weights)
    # model.feature_extractor_mode()
    #unit test for feature size
    testing_size = TestFeatureSize(model, feature_size) # this will confirm that the feature size is correct
    assert model is not None, "Failed to load the model"
    try:
        testing_size.test_feature_size()
        print(f"The loaded model under evaluation is in indeed with {feature_size} feature size!")

    except AssertionError as e:
        # add an error message to the assertion error
        e.args += (f"Expected feature size {feature_size}, got {model.features_out.in_features}")   
        raise e # if the feature size is not correct, raise an error
    

    return model


def main_resnet_pca():
    # Set seed for reproducibility
    seed_value = 42  # can be any integer value
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    random.seed(seed_value)

    #### set device #####
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')  # Default to '0' if not set
    print(f"GPU ID: {gpu_id}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    #####################
    ###### prep directories ######
    data_root ="/media/alabutaleb/data/cub200/"

    # load_dir = f"/home/alabutaleb/Desktop/confirmation/baselines_allsizes/weights"   
    load_dir = f"/home/alabutaleb/Desktop/confirmation"    
    # pca_weights = f"/home/alabutaleb/Desktop/confirmation/pca_weights"
    # ensure_directory_exists(pca_weights)
    pca_logs = f"/home/alabutaleb/Desktop/confirmation/kd_logs"
    # log_save_path = f"/home/alabutaleb/Desktop/confirmation/"
    log_save_folder_kd = os.path.join(pca_logs, f"logs_gpu_{gpu_id}")
    os.makedirs(log_save_folder_kd, exist_ok=True)
    ensure_directory_exists(log_save_folder_kd)

    print(f"Weights will be loaded from: {load_dir}")
    # print(f"Weight files will be saved in: {save_dir}")
    print(f"Log files for this KD experiment will be saved in: {log_save_folder_kd}")
    #####################
   
    #####################
    print("#"*30)
    print(f"You are curently using {device} device")
    print("#"*30)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    #### hyperparameters #####
    batch_size  = 256
    warmup_epochs=20 
    lr=0.00007 # best for baseline experiments
 
    weight_decay = 2e-05
    whiten = False
    # whiten = True

    # compression_level= float(compressed_features_size/n_components)


    DEBUG_MODE = False
    use_early_stopping=True

    #### get data #####root, batch_size=32,num_workers=10   
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    num_classes_cub200 = dataloadercub200.get_number_of_classes()

    if DEBUG_MODE:
        # Beware that if in debug_mode then batch size will always be 32
        # create small subset of data to make debugging faster
        trainloader_cub200_dump, testloader_cub200_dump = dataloadercub200.get_dataloaders()
        trainloader_cub200, testloader_cub200, batch_size = create_subset_data(trainloader_cub200_dump, testloader_cub200_dump, batch_size=32)
        epochs = 5
        T_max=int(epochs/2)
        last_epoch = -1
    else:
        trainloader_cub200, testloader_cub200 = dataloadercub200.get_dataloaders()
        epochs = 40
        T_max=int(epochs/2)
        last_epoch = -1


   
    # > Logic:
    # 1. Load vanilla
    # 2. Set feature extractor mode to remove last layer AND to freeze all the weights
    # 3. Pass trainloader through the model.
    # 4. Fit via PCA
    # 5. Return PCA
    # 6. Pass testloader through the model
    # 7. Now, train and evaluate fc layer
    # 8. Fit the new features (generated from testLoader) to the PCA
    # 9. Now pass these new features via the fc layer and evaluate classification performance

  
    criterion = nn.CrossEntropyLoss()

    # TODO send logs directory
    # TODO send save directory for resnet after linear probing
    

    # test on retrieval

    # test on retrieval with PCA

    # test on retrieval with PCA and whitening








if __name__ == "__main__":
    main_resnet_pca()

