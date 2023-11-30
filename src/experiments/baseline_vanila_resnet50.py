

import sys
ROOT="/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp"
sys.path.append(f"{ROOT}/src")
print(sys.path)
import torch
import torch.nn as nn
import os
import numpy as np
import random
from exp_logging.metricsLogger import MetricsLogger

from torchvision.models import resnet50, ResNet50_Weights
from dataloaders.cub200loader import DataLoaderCUB200
from models.resnet50_vanilla import ResNet50_vanilla
from models.resnet50_conv_compressionV2 import ResNet50_convV2, ResNet50_convV3_BN

from optimizers.adam_lr_var_test import AdamOptimizerVar
from schedulers.cosine import CosineAnnealingLRWrapperWithWarmup
from trainers.vanilla_trainer_resnet50 import ResnetTrainer_test
from trainers.resnets_trainer import ResnetTrainer

from loss.ce import CustomCrossEntropyLoss

import matplotlib.pyplot as plt
from utils.similarities import evaluate_on_retrieval
from utils.features_unittest import TestFeatureSize
from utils.debugging_functions import create_subset_data



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
   
    def create_optimizer_var_lr(model, lr, weight_decay):        
        optimizer = AdamOptimizerVar(model, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        return optimizer.get_optimizer()

    
    def create_scheduler_cosw(optimizer, T_max=100, warmup_epochs=20, warmup_decay="cosine"):
        scheduler = CosineAnnealingLRWrapperWithWarmup(optimizer, T_max=100, warmup_epochs=20, warmup_decay="cosine")
        return scheduler    
    


    def train_and_evaluate(dataset_name, model, optimizer, scheduler, lr,
                           criterion, trainloader, testloader, 
                           use_early_stopping, device, num_classes, metrics_logger,
                           batch_size, epochs, feature_size, save_dir,log_save_path):
        
        print(f"Training with {dataset_name} dataset and feature size: {feature_size}")
        trainer = ResnetTrainer(model, optimizer, criterion, lr, scheduler, trainloader, 
                                testloader, feature_size, use_early_stopping,
                                device, num_classes, log_save_path, metrics_logger, epochs=epochs, 
                                dataset_name=dataset_name)
        
        model, training_loss, training_accuracy, average_validation_loss, average_validation_accuracy = trainer.train_model()
        trainer.plot_metrics(training_loss, average_validation_loss, training_accuracy, average_validation_accuracy)

        torch.save(model.state_dict(), f'{save_dir}/resnet50_feature_size_{feature_size}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth')
        print(f"Model saved at {save_dir}/resnet50_feature_size_{feature_size}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth")

        return model, training_loss, training_accuracy, average_validation_loss, average_validation_accuracy
    
    
    #### set device #####
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')  # Default to '0' if not set
    print(f"GPU ID: {gpu_id}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
    #####################
    #### directory to save fine tuned weights #####
    base_directory = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned"
    # weight_save_path =  "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned"
    data_root ="/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/cub200/"


    # Construct the dynamic directory path
    load_dir = f"/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiment_gpu_{gpu_id}/weights"
    # load_dir = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiment_gpu_1/weights"
    #####################
    # base_directory = "./experiments_results"
    # unique_directory = os.path.join(base_directory, f"experiment_gpu_{gpu_id}")
    save_dir_folder = os.path.join(base_directory, f"experiment_gpu_{gpu_id}")
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

            
    #### hyperparameters #####
    batch_size  = 256
    lr = 0.00007  # 0.00007 is the best for both vanilla and modified resnet50. on vanilla it yields 82.9651% accuracy on test set
    
    dataset_names= "cub200"


    weight_decay= 2e-05
    # weight_decay= 1e-5
    DEBUG_MODE = False
    use_early_stopping=True

    warmup_epochs=20 
    lr_warmup_decay=0.01 
    # lr_warmup_decay=0.3

 
    #### get data #####root, batch_size=32,num_workers=10   
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    num_classes_cub200 = dataloadercub200.get_number_of_classes()

    if DEBUG_MODE:
        # create small subset of data to make debuggin faster
        trainloader_cub200_dump, testloader_cub200_dump = dataloadercub200.get_dataloaders()
        trainloader_cub200, testloader_cub200, batch_size = create_subset_data(trainloader_cub200_dump, testloader_cub200_dump, batch_size=32)
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
    print(f"You are using batch size: {batch_size}")
    print(f"You are using epochs: {epochs}")
    print(f"With Learning rate: {lr}")
    print(f"You are using early stopping: {use_early_stopping}")
    print("/\\"*30)

    # num_classes_cub200 = dataloadercub200.get_number_of_classes()    

    model = ResNet50_vanilla(num_classes_cub200, set_eval_mode=False, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)
    model.fine_tune_mode()

    model = model.to(device)
    criterion = CustomCrossEntropyLoss()
    optimizer  = create_optimizer_var_lr(model, lr, weight_decay=weight_decay)   
    scheduler_var = create_scheduler_cosw(optimizer, T_max, warmup_epochs, warmup_decay="cosine") 

    print(f"The model is on device: {next(model.parameters()).device}")

    metrics_logger = MetricsLogger()

    feature_size = "xxxxx" # vanilla case
    # trainer = ResnetTrainer(model, optimizer, criterion, lr, scheduler_var, trainloader_cub200, 
    #                         testloader_cub200, feature_size, use_early_stopping,
    #                         device, num_classes_cub200, log_save_path, metrics_logger, epochs=epochs, 
    #                         dataset_name=dataset_names)    
    
    # trainer = ResnetTrainer_test(model, criterion, optimizer, num_classes_cub200, device, use_early_stopping, scheduler_var)    


    model_cub200, train_loss_cub200, training_accuracy_cub200, val_loss_cub200, val_acc_cub200 = train_and_evaluate(dataset_names, model, optimizer, scheduler_var, lr,
                                                                    criterion, trainloader_cub200, 
                                                                    testloader_cub200, use_early_stopping,
                                                                    device, num_classes_cub200, metrics_logger,
                                                                    batch_size, epochs, feature_size, save_dir,log_save_path)  

    # fine_tuned_model = trainer.train(trainloader_cub200, testloader_cub200, epochs)
    
    # metrics_logger.log_and_calculate_metrics(dataset_name, feature_size, model, testloader, last_training_loss, device)



    # After all experiments, plot the metrics using the metrics logger
    # metrics_logger.plot_metrics()
    
    # Plot the training and validation losses and accuracies
    # fig1 = trainer.plot_loss_vs_epoch()
    # fig2 = trainer.plot_acc_vs_epoch()
    # fig1.savefig(f'vanila_WITH_2048_CONV_test_loss_vs_epoch_scheduler_Adam_cos_warmup_batch_size_{batch_size}_lr_{lr}_epochs_{epochs}_with_dropout2.png')
    # fig2.savefig(f'vanila_WITH_2048_CONV_test_acc_vs_epoch_scheduler_Adam_cos_warmup_batch_size_{batch_size}_lr_{lr}_epochs_{epochs}_with_dropout2.png')    



if __name__ == "__main__":
    main_resnet_test()

