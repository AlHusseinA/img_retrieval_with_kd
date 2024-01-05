
import sys
import os

ROOT="/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp"
sys.path.append(f"{ROOT}/src")
print(f"{sys.path=}")
print(f"{os.getcwd()=}")
import torch
import torch.nn as nn
import torch.nn.init as init
from loss.kd_loss import DistillKL
from loss.ce import CustomCrossEntropyLoss
from schedulers.cosine import CosineAnnealingLRWrapper, CosineAnnealingLRWrapperWithWarmup
from optimizers.adam_for_fc import AdamOptimizerFC
# from exp_logging.metricsLogger import MetricsLogger
from exp_logging.metricsLogger_kd import MetricsLoggerKD
from exp_logging.metricsLogger import MetricsLogger
from torchvision.models import resnet50, ResNet50_Weights
from datetime import datetime
import yaml
from dataloaders.cub200loader import DataLoaderCUB200
# import models.resnet50 as resnet50
from models.resnet50_conv_compressionV2 import ResNet50_convV2
from models.resnet50_vanilla import ResNet50_vanilla

import matplotlib.pyplot as plt
from optimizers.adam_lr_var_test import AdamOptimizerVar

from utils.similarities import evaluate_on_retrieval
from utils.features_unittest import TestFeatureSize
from utils.debugging_functions import create_subset_data

from utils.helper_functions import check_size
from utils.helpers_function_kd import plot_performance
# from trainers.train_eval_fc_pca import train_eval_fc_pca
from trainers.kd_trainer import KnowledgeDistillationTrainer
import numpy as np
import random

from copy import deepcopy

# for debugging
from torch.utils.data import Subset


def create_optimizer_var_lr(model, lr, weight_decay):
    # Check if the model is wrapped in DataParallel
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    temp = AdamOptimizerVar(model, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # Now you can access model.features
    optimizer = temp.get_optimizer()
    optimizer.actual_optimizer_name = type(optimizer).__name__
    return optimizer

def create_scheduler_cosw(optimizer, T_max=100, warmup_epochs=20, warmup_decay="cosine"):
    scheduler = CosineAnnealingLRWrapperWithWarmup(optimizer, T_max=100, warmup_epochs=20, warmup_decay="cosine")
    scheduler.actual_scheduler_name = type(scheduler).__name__
    return scheduler 

def ensure_directory_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory {directory_path} created.")
    else:
        print(f"Directory {directory_path} already exists.")



def load_resnet50_convV2(num_classes_cub200,feature_size, dataset_name, batch_size, lr, load_dir):
    model = ResNet50_convV2(feature_size, num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)        
    # file name = resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e-05.pth
    fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_{feature_size}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth')
    model.load_state_dict(fine_tuned_weights)
    # model.feature_extractor_mode()
    #unit test for feature size
    test = deepcopy(model)
    testing_size = TestFeatureSize(test, feature_size) # this will confirm that the feature size is correct
    try:
        testing_size.test_feature_size()
        print(f"The loaded model under evaluation is in indeed with {feature_size} feature size! This is the correct vanilla size.")

    except AssertionError as e:
        # add an error message to the assertion error
        e.args += (f"Expected feature size {feature_size}, got {test.features_out.in_features}")   
        raise e # if the feature size is not correct, raise an error
    return model

def load_resnet50_unmodifiedVanilla(num_classes_cub200,feature_size, dataset_name, batch_size, lr, load_dir):

    model = ResNet50_vanilla(num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)        
    fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e.pth')
    # fine_tuned_weights = torch.load(f'{load_dir}/resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e.pth')

    model.load_state_dict(fine_tuned_weights)
    # model.feature_extractor_mode()
    #unit test for feature size
    test = deepcopy(model)
    testing_size = TestFeatureSize(test, feature_size) # this will confirm that the feature size is correct

    assert model is not None, "Failed to load the model"
    try:
        testing_size.test_feature_size()
        print(f"The loaded model under evaluation is in indeed with {feature_size} feature size!")

    except AssertionError as e:
        # add an error message to the assertion error
        e.args += (f"Expected feature size {feature_size}, got {test.features_out.in_features}")   
        raise e # if the feature size is not correct, raise an error
    

    return model


def main_kd():
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

    load_dir_baseline = f"/home/alabutaleb/Desktop/confirmation/baselines_allsizes/weights"   

    load_dir_vanilla = f"/home/alabutaleb/Desktop/confirmation"    # "resnet50_feature_size_vanilla_cub200_batchsize_256_lr_7e.pth"
    kd_logs = f"/home/alabutaleb/Desktop/confirmation/kd_logs"

    ks_weights_save = f"/home/alabutaleb/Desktop/confirmation/kd_weights"
    # log_save_path = f"/home/alabutaleb/Desktop/confirmation/"
    log_save_folder_kd = os.path.join(kd_logs, f"logs_gpu_{gpu_id}")
    ks_weights_save = os.path.join(ks_weights_save, f"logs_gpu_{gpu_id}")

    os.makedirs(log_save_folder_kd, exist_ok=True)
    os.makedirs(ks_weights_save, exist_ok=True)
    ensure_directory_exists(log_save_folder_kd)
    ensure_directory_exists(ks_weights_save)

    print(f"Vanilla weights will be loaded from: {load_dir_vanilla}")
    print(f"Baseline weights will be loaded from: {load_dir_baseline}")
    print(f"Weight files for this KD pipeline will be saved in: {ks_weights_save}")
    print(f"Log files for this KD experiment will be saved in: {log_save_folder_kd}")
    #####################
    dataset_name = "cub200"

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
    # lr = 0.001
    weight_decay = 2e-05


    DEBUG_MODE = False
    # DEBUG_MODE = True

    use_early_stopping=True

    #### get data #####root, batch_size=32,num_workers=10   
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    num_classes_cub200 = dataloadercub200.get_number_of_classes()

    if DEBUG_MODE:
        # create small subset of data to make debugging faster
        trainloader_cub200_dump, testloader_cub200_dump = dataloadercub200.get_dataloaders()
        trainloader_cub200, testloader_cub200, batch_size = create_subset_data(trainloader_cub200_dump, testloader_cub200_dump, batch_size=32)
        # trainloader_cub200, testloader_cub200, batch_size = create_subset_data2(trainloader_cub200_dump, testloader_cub200_dump, subset_ratio=0.1, batch_size=32)
        epochs = 25
        T_max=20
        last_epoch = -1
    else:
        trainloader_cub200, testloader_cub200 = dataloadercub200.get_dataloaders()
        epochs = 1000 #1000
        T_max=int(epochs/2)
        last_epoch = -1


    metrics_logger_van = MetricsLogger()



    feature_size_unmodifed = 2048
    # feature_size_student = [8, 32, 64, 128, 256, 512, 1024, 2048]
    # feature_size_students = [8, 32, 64, 128, 1024]
    # feature_size_students = [32, 64, 1024]
    feature_size_students = [8, 16, 128, 256, 512 ]
    # feature_size_students = [8]

    if DEBUG_MODE:
        batch_size = 256 

    for feature_size_student in feature_size_students:
   
        teacher_model = load_resnet50_unmodifiedVanilla(num_classes_cub200, feature_size_unmodifed, "cub200", batch_size, lr, load_dir_vanilla)
        # student_model = load_resnet50_convV2(num_classes_cub200, feature_size_student, "cub200", batch_size, lr, load_dir_baseline) # if starting from finetuned weights
        student_model = ResNet50_convV2(feature_size_student, num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None) # if starting from imagenet weights

        output_size_teacher = check_size(teacher_model)
        output_size_student = check_size(student_model)
        print(f"Feature layer size of Teacher is: {output_size_teacher}")
        print(f"Feature layer size of Student is: {output_size_student}")

        if output_size_teacher != 2048:
            print(f"Feature layer size of Teacher is: {check_size(teacher_model)=}")
            raise ValueError("The teacher model is not an unmodified resnet50") 
        if output_size_student != feature_size_student:
            raise ValueError("The student model is not with the desired feature size")


        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            teacher_model = nn.DataParallel(teacher_model)
            student_model = nn.DataParallel(student_model)

        teacher_model.to(device)
        student_model.to(device)    

        optimizer_student = create_optimizer_var_lr(student_model, lr, weight_decay)
        scheduler_student = create_scheduler_cosw(optimizer_student, T_max, warmup_epochs=20, warmup_decay="cosine")
        criterion = CustomCrossEntropyLoss()
        distill_loss = DistillKL(criterion, T=1, alpha=0.5) # T=3, T=0.01, alpha=0.6
        logger = MetricsLoggerKD()
        kd_student = KnowledgeDistillationTrainer(teacher_model, student_model, criterion, distill_loss, optimizer_student, scheduler_student, logger, num_classes_cub200, lr, device, log_save_folder_kd, use_early_stopping, temperature=3)

        trained_student = kd_student.train(trainloader_cub200, testloader_cub200, epochs)

        # def plot_performance(log_save_path, mode, student_size=None):

        plot_performance(log_save_folder_kd, student_size=feature_size_student)


        # When saving the model:
        if isinstance(student_model, torch.nn.DataParallel):
            torch.save(student_model.module.state_dict(), f'{ks_weights_save}/KD_student_resnet50_feature_size_{feature_size_student}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth')
        else:
            torch.save(student_model.state_dict(), f'{ks_weights_save}/KD_student_resnet50_feature_size_{feature_size_student}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth')

    metrics_logger_van.plot_multiple_metrics(log_save_folder_kd, feature_size_student, lr, dataset_name)
    
    # print(f"Model saved at {ks_weights_save}/KD_student_resnet50_feature_size_{feature_size}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth")
  
  

    # TODO send logs directory
    # TODO send save directory for resnet after linear probing
 






if __name__ == "__main__":
    main_kd()

