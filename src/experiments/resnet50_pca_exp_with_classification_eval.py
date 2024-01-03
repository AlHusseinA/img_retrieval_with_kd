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
from loss.ce import CustomCrossEntropyLoss

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
    pca_weights = f"/home/alabutaleb/Desktop/confirmation/pca_weights"
    ensure_directory_exists(pca_weights)
    pca_logs = f"/home/alabutaleb/Desktop/confirmation/pca_logs"
    # log_save_path = f"/home/alabutaleb/Desktop/confirmation/"
    log_save_folder = os.path.join(pca_logs, f"logs_gpu_{gpu_id}")
    os.makedirs(log_save_folder, exist_ok=True)

    print(f"Weights will be loaded from: {load_dir}")
    # print(f"Weight files will be saved in: {save_dir}")
    print(f"Log files for this PCA experiment will be saved in: {log_save_folder}")
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
    # lr=0.00007 # best for baseline experiments
    # lr=0.0005
    # lr=0.005
    # lr=0.001
    lr=0.01

    n_components=2048
    compressed_features_size=128 
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


   
    #### print hyperparameters #####
    print("/\\"*30)
    print(f"You are using batch size: {batch_size}")
    # print(f"You are compressing features from {n_components} dimensions to {compressed_features_size} dimensions size. Compression level is: {compression_level}")
    print(f"You are using epochs: {epochs}")
    print(f"With Learning rate: {lr}")
    print(f"You are using early stopping: {use_early_stopping}")
    print("######"*30)
    print(f"During finetuning with the presence of a pca 'layer', max pca compression would be from 2049 to 200 which is the number of classes in cub200 dataset")


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

    # Steps 1, 2, 3, 4, and 5
    features_extractor = load_resnet50_unmodifiedVanilla(num_classes_cub200,n_components, "cub200", batch_size, lr, load_dir)
                        #  load_resnet50_unmodifiedVanilla(num_classes_cub200,feature_size, dataset_name, batch_size, lr, load_dir):

    features_extractor.feature_extractor_mode()          # to remove the last layer
    features_extractor.to(device)
    # fit the pca with the training data
    pca, train_features = pca_fit_features(features_extractor, trainloader_cub200, n_components=num_classes_cub200, whiten=whiten, device=device)

    # Assert that PCA was successful
    assert pca is not None, "PCA fitting failed"
    assert hasattr(pca, 'components_'), "PCA fitting did not compute components"
    assert pca.n_components_ <= num_classes_cub200, "PCA components exceed the specified number"

    # Now pass testloader via the feature extractor and save the results in a list
    test_features = generate_and_process_features(features_extractor, testloader_cub200, device)
    # train_features = generate_and_process_features(features_extractor, trainloader_cub200, device)

    # print(f"Shape of test_featuers: {len(test_features)}")
    # print(f"shape of first item of test_features is: {test_features[0].shape}")
    # print(f"Type of pca: {type(pca)}")
    if isinstance(test_features, list):
        # Assuming each element in test_features is a numpy array
        test_features = np.concatenate(test_features, axis=0)
        print(f"Shape of test_features: {test_features.shape}")

    # print(f"size of the testLoader is: {len(testloader_cub200.dataset)}")

    # Before transforming in main
    if test_features.ndim != 2:
        raise ValueError(f"test_features must be 2D, got {test_features.ndim}D")
    # compress those features
    compressed_test_featues = pca.transform(test_features)
    compressed_trainfeatues = pca.transform(train_features)

    print(f"Shape of compressed_test_features: {compressed_test_featues.shape}")
    # assert compressed_features_size.shape[1] == num_classes_cub200, "Size of compressed featuers MUST equal the number of classes"

    # Define a new fc layer and loss
    new_fc = nn.Linear(in_features=200, out_features=200)  # Assuming the feature size is 2048
    init.xavier_uniform_(new_fc.weight)

    criterion = CustomCrossEntropyLoss()

    # test on classification
    optimizer  = create_optimizer_fc(new_fc, lr, weight_decay=weight_decay)   
    trained_fc_layer = fc_trainer(new_fc, epochs, optimizer, criterion, trainloader_cub200, testloader_cub200, compressed_trainfeatues, compressed_test_featues, device)

    # TODO send logs directory
    # TODO send save directory for resnet after linear probing
    

    # test on retrieval
    # ADD RETRIEVAL CODE
    
    # test on retrieval with PCA

    # test on retrieval with PCA and whitening








if __name__ == "__main__":
    main_resnet_pca()

