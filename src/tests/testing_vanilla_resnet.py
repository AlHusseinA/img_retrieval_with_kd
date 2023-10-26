

import sys
ROOT="/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp"
sys.path.append(f"{ROOT}/src")
print(sys.path)
import torch
import torch.nn as nn
import os
# from dataloaders.cifar10 import DataLoaderCIFAR10
from torchvision.models import resnet50, ResNet50_Weights
from dataloaders.cub200loader import DataLoaderCUB200
from models.resnet50_vanilla import ResNet50_vanilla
from optimizers.sgd_var_lr_test import SGDOptimizerVariableLR
from optimizers.adam import AdamOptimizer
from optimizers.adam_lr_var_test import AdamOptimizerVar
from schedulers.cosine import CosineAnnealingLRWrapper, CustomCosineAnnealingWarmUpLR, CosineScheduler
from trainers.vanilla_trainer_resnet50 import ResnetTrainer_test
from loss.ce import CustomCrossEntropyLoss

import matplotlib.pyplot as plt
from utils.similarities import evaluate_on_retrieval
from utils.features_unittest import TestFeatureSize
from utils.debugging_functions import create_subset_data
import numpy as np
import random


def main_resnet_test():
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

  
    def create_optimizer(model, lr,weight_decay):        
        optimizer = AdamOptimizer(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        return optimizer.get_optimizer()
    
    def create_optimizer_var_lr(model, lr,weight_decay):        
        optimizer = AdamOptimizerVar(model, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        return optimizer.get_optimizer()
    # def create_optimizer_sgd(model, lr,weight_decay):
    #     optimizer = SGDOptimizer(model, lr=lr, momentum=0.9, weight_decay=weight_decay)
    #     return optimizer.get_optimizer()
    
    def create_optimizer_var_sgd(model, lr,weight_decay):
        optimizer = SGDOptimizerVariableLR(model, lr=lr, momentum=0.9, weight_decay=weight_decay)
        return optimizer.get_optimizer()
    
    def create_scheduler_cos(optimizer, T_max,  eta_min=0.01):
        scheduler = CosineAnnealingLRWrapper(optimizer, T_max, eta_min=eta_min)
        return scheduler
    
    def create_scheduler_cosw(optimizer, lr_warmup_epochs, lr_warmup_decay, T_max, last_epoch=-1):
        scheduler = CustomCosineAnnealingWarmUpLR(optimizer, lr_warmup_epochs, lr_warmup_decay, T_max, last_epoch)
        return scheduler    
    
    
    #### set device #####
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')  # Default to '0' if not set
    print(f"GPU ID: {gpu_id}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    #####################
    # Define the conditional variable for test/debugging
    base_directory = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned"
    # weight_save_path =  "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned"
    data_root ="/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/cub200/"
    load_dir = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiment_gpu_1/weights"


    print("Base Directory:", base_directory)
    print("Data Root:", data_root)
    print("Load Directory:", load_dir)
    #####################
 
    print("#"*30)
    print(f"You are curently using {device} device")
    print("#"*30)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
 
    # Create dictionaries to store metrics
    metrics_cub200_finetuning = {}
    # metrics_cub200_retrieval = {}

           
    #### hyperparameters #####
    batch_size  = 256
    # epochs = 1000 #1 # 500
    epochs = 1000
    lr=0.00001
    # lr=0.5
    # lr=0.1
    print(f"Learning rate: {lr}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {epochs}")


    weight_decay=2e-05
    # weight_decay= 1e-5
    DEBUG_MODE = False
    use_early_stopping=True

    # use_early_stopping=True
    lr_warmup_epochs=20 
    lr_warmup_decay=0.01 
    # lr_warmup_decay=0.3

    T_max=int(epochs/2)
    last_epoch = -1
    #### print hyperparameters #####
    print("/\\"*30)
    print(f"You are using batch size: {batch_size}")
    print(f"You are using epochs: {epochs}")
    print(f"You are using early stopping: {use_early_stopping}")
    print("/\\"*30)
    

    #### get data #####root, batch_size=32,num_workers=10   
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    num_classes_cub200 = dataloadercub200.get_number_of_classes()

    if DEBUG_MODE:
        trainloader_cub200_dump, testloader_cub200_dump = dataloadercub200.get_dataloaders()
        trainloader_cub200, testloader_cub200 = create_subset_data(trainloader_cub200_dump, testloader_cub200_dump, batch_size=32)
    else:
        trainloader_cub200, testloader_cub200 = dataloadercub200.get_dataloaders()

    


 
    model = ResNet50_vanilla(num_classes_cub200, set_eval_mode=False, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)
    model = model.to(device)
    criterion = CustomCrossEntropyLoss()
    # optimizer = create_optimizer_var_lr(model, lr, weight_decay=weight_decay)
    optimizer  = create_optimizer_var_lr(model, lr, weight_decay=weight_decay)
    # optimizer = create_optimizer_var_sgd(model, lr, weight_decay=weight_decay)
    # optimizer   = create_optimizer(model, lr, weight_decay=weight_decay)


    # scheduler = create_scheduler_cos(optimizer, epochs, lr)
    scheduler_var = create_scheduler_cos(optimizer, T_max,  eta_min=0)
    # scheduler_var = create_scheduler_cosw(optimizer, lr_warmup_epochs, lr_warmup_decay, T_max, last_epoch=-1)    


    # scheduler = CosineScheduler(20, warmup_steps=5, base_lr=0.3, final_lr=0.01)
    # scheduler_var = CosineScheduler(optimizer, max_update=20, base_lr=0.8, final_lr=0.01)
    # d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])# d2l.plot(torch.arange(num_epochs), [scheduler(t) for t in range(num_epochs)])

    print(f"The model is still on device: {next(model.parameters()).device}")

    trainer = ResnetTrainer_test(model, criterion, optimizer, device, use_early_stopping, scheduler_var)    
    fine_tuned_model = trainer.train(trainloader_cub200, testloader_cub200, epochs)
    # Plot the training and validation losses and accuracies
    fig1 = trainer.plot_loss_vs_epoch()
    fig2 = trainer.plot_acc_vs_epoch()
    fig1.savefig(f'vanila_test_loss_vs_epoch_scheduler_Adam_lr_cos_batch_size_{batch_size}_lr_{lr}_epochs_{epochs}.png')
    fig2.savefig(f'vanila_test_acc_vs_epoch_scheduler_Adam_lr__cos_batch_size_{batch_size}_lr_{lr}_epochs_{epochs}.png')    



if __name__ == "__main__":
    main_resnet_test()

