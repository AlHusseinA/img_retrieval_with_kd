import sys
ROOT="/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp"
sys.path.append(f"{ROOT}/src")
print(sys.path)
import torch
import torch.nn as nn
import os
# from dataloaders.cifar10 import DataLoaderCIFAR10
from exp_logging.metricsLogger import MetricsLogger
from torchvision.models import resnet50, ResNet50_Weights
from datetime import datetime
import yaml
from dataloaders.cub200loader import DataLoaderCUB200
# from utils.metrics import calculate_map, calculate_recall_at_k
import models.resnet50 as resnet50
from optimizers.sgd import SGDOptimizer, SGDOptimizerVariableLR
from optimizers.adam import AdamOptimizer
from optimizers.adam_lr_var import AdamOptimizerVar
from schedulers.cosine import CosineAnnealingLRWrapper, CosineAnnealingLRWrapperWithWarmup
from trainers.resnets_trainer import ResnetTrainer
from loss.ce import CustomCrossEntropyLoss
import matplotlib.pyplot as plt
# from utils.gallery import extract_gallery_features
from utils.similarities import evaluate_on_retrieval
from utils.features_unittest import TestFeatureSize
from utils.debugging_functions import create_subset_data
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

  
    def create_optimizer(model, lr,weight_decay):        
        temp = AdamOptimizer(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer = temp.get_optimizer()
        optimizer.actual_optimizer_name = type(optimizer).__name__
        return optimizer    

    
    def create_optimizer_var_lr(model, lr,weight_decay):        
        temp = AdamOptimizerVar(model, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer = temp.get_optimizer()
        optimizer.actual_optimizer_name = type(optimizer).__name__
        return optimizer 
    
    # def create_optimizer_sgd(model, lr,weight_decay):
    #     optimizer = SGDOptimizer(model, lr=lr, momentum=0.9, weight_decay=weight_decay)
    #     return optimizer.get_optimizer()
    
    def create_optimizer_var_sgd(model, lr,weight_decay):
        temp = SGDOptimizerVariableLR(model, lr=lr, momentum=0.9, weight_decay=weight_decay)
        optimizer = temp.get_optimizer()
        optimizer.actual_optimizer_name = type(optimizer).__name__
        return optimizer 
    
    def create_scheduler_cos(optimizer, T_max,  eta_min=0.01):
        scheduler = CosineAnnealingLRWrapper(optimizer, T_max, eta_min=eta_min)
        scheduler.actual_scheduler_name = type(scheduler).__name__
        return scheduler 
    
    def create_scheduler_cosw(optimizer, T_max=100, warmup_epochs=20, warmup_decay="cosine"):
        scheduler = CosineAnnealingLRWrapperWithWarmup(optimizer, T_max=100, warmup_epochs=20, warmup_decay="cosine")
        scheduler.actual_scheduler_name = type(scheduler).__name__
        return scheduler        
    

    def train_and_evaluate(dataset_name, model, optimizer, scheduler, lr,
                           criterion, trainloader, testloader, 
                           use_early_stopping, device, num_classes, metrics_logger,
                           epochs, feature_size, save_dir,log_save_path):
        
        print(f"Training with {dataset_name} dataset and feature size: {feature_size}")
        trainer = ResnetTrainer(model, optimizer, criterion, lr, scheduler, trainloader, 
                                testloader, feature_size, use_early_stopping,
                                device, num_classes, log_save_path, metrics_logger, epochs=epochs, 
                                dataset_name=dataset_name)
        
        model, training_loss, training_accuracy, average_validation_loss, average_validation_accuracy = trainer.train_model()
        trainer.plot_metrics(training_loss, average_validation_loss, training_accuracy, average_validation_accuracy)

        torch.save(model.state_dict(), f'{save_dir}/resnet50_{feature_size}_{dataset_name}_batchsize_{batch_size}_lr_{lr}.pth')
        print(f"Model saved at {save_dir}/resnet50_{feature_size}_{dataset_name}_fine_tuned.pth")
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
    # load_dir = f"/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiment_gpu_{gpu_id}/weights"
    load_dir = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiment_gpu_1/weights"
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
 
    # Create dictionaries to store metrics
    metrics_cub200_finetuning = {}
    metrics_cub200_retrieval = {}

    ensure_directory_exists(save_dir)
           
    #### hyperparameters #####
    batch_size  = 256
    epochs = 7500
    # epochs = 750
    lr=0.00001      # lr=0.5
    # lr=0.1

    weight_decay=2e-05

    DEBUG_MODE = False
    use_early_stopping=True

    # use_early_stopping=True
    warmup_epochs=20 
    lr_warmup_decay=0.01 

    T_max=int(epochs/2)
    last_epoch = -1

    #### print hyperparameters #####
    print("/\\"*30)
    print(f"Learning rate: {lr}")
    print(f"You are in DEBUG MODE: {DEBUG_MODE}")
    print(f"You are using batch size: {batch_size}")
    print(f"You are using epochs: {epochs}")
    print(f"You are using EARLY STOPPING: {use_early_stopping}")
    print("/\\"*30)
    

    #### get data #####root, batch_size=32,num_workers=10   
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    num_classes_cub200 = dataloadercub200.get_number_of_classes()

    if DEBUG_MODE:
        # create small subset of data to make debuggin faster
        trainloader_cub200_dump, testloader_cub200_dump = dataloadercub200.get_dataloaders()
        trainloader_cub200, testloader_cub200, batch_size = create_subset_data(trainloader_cub200_dump, testloader_cub200_dump, batch_size=32)
    else:
        trainloader_cub200, testloader_cub200 = dataloadercub200.get_dataloaders()
    

    #### prep logger ####
    
    # Initializing metrics logger for later use in logging metrics
    dataset_names=["cub200"]

    metrics_logger = MetricsLogger()
    # metrics_logger.to(device)

    #### get loss for fine tuning #####
    criterion = CustomCrossEntropyLoss()

    #### feature sizes #####
    # feature_sizes = [16]
    # feature_sizes = [2048]
    # feature_sizes = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    feature_sizes = [2048, 1024, 512, 256, 128, 64, 32, 16, 8]
    # feature_sizes = [16]#, 32, 64, 128, 256, 512, 1024, 2048]




    # Get current date and time

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    scheduler = None
    scheduler_var = None
    optimizer = None
    optimizer_var = None
    optimizer_var_sgd = None
    optimizer_sgd = None
    flag_scheduler = 0
    flag_optimizer = 0
    #### train model #####
    # Main loop to iterate over each feature size
    for feature_size in feature_sizes:           
        # print(f"Training model with feature size: {feature_size}")  
        print(f"Fine-tuning ResNet50s with feature sizes: {feature_sizes}, starting with feature size: {feature_size}\n")
        # #### get model ##### 
        model = resnet50.ResNet50(feature_size, num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)
        model.fine_tune_mode()
        model = model.to(device)
        # optimizer  = create_optimizer(model, lr, weight_decay=weight_decay)
        optimizer  = create_optimizer_var_lr(model, lr, weight_decay=weight_decay)
        scheduler = create_scheduler_cosw(optimizer, T_max, warmup_epochs, warmup_decay="cosine") 

        # optimizer_var = create_optimizer_var_lr(model, lr, weight_decay=weight_decay)
        # optimizer_var_sgd = create_optimizer_var_sgd(model, lr, weight_decay=weight_decay)
        # optimizer_sgd = create_optimizer_sgd(model, lr, weight_decay=weight_decay)
        # TODO
        # add SGD optimizer


        # scheduler = create_scheduler_cos(optimizer, epochs, lr)
        # scheduler_var = create_scheduler_cosw(optimizer_sgd, lr_warmup_epochs, lr_warmup_decay, T_max, last_epoch=-1)
                    
        # add config for which resnet you used, decay lr or not


        model_cub200, train_loss_cub200, training_accuracy_cub200, val_loss_cub200, val_acc_cub200 = train_and_evaluate("CUB-200", model, optimizer, scheduler, lr,
                                                                      criterion, trainloader_cub200, 
                                                                      testloader_cub200, use_early_stopping,
                                                                      device, num_classes_cub200, metrics_logger,
                                                                      epochs, feature_size, save_dir,log_save_path)  
              
        metrics_cub200_finetuning[feature_size] = {'train_loss': train_loss_cub200, 'train_acc': training_accuracy_cub200, 
                                        'val_loss': val_loss_cub200, 'val_acc': val_acc_cub200}
        
        # print(f"\nClassification validation accuracy for CUB-200: {metrics_cub200_finetuning[feature_size]['val_acc']}")
        print(f"Finetuning model with feature size: {feature_size} is complete!\n") 
        print("#"*30)

    #retrieval
    metrics_logger_retrieval = MetricsLogger() 



    #### set the confi dict #####
    # Initialize config dictionary
    config_dict = {}
 
    if scheduler_var is not None:
        config_dict["warmup_epochs"] = warmup_epochs
        config_dict["lr_warmup_decay"] = lr_warmup_decay
        config_dict["T_max"] = T_max
        config_dict["last_epoch"] = last_epoch
        flag_scheduler = 1
    else:   
        config_dict["scheduler"] = "None"
        flag_scheduler = 1

    if optimizer_var is not None and flag_optimizer == 0:
        config_dict["optimizer"] = "AdamOptimizerVar"
        flag_optimizer = 1
    elif optimizer_var_sgd is not None and flag_optimizer == 0:
        config_dict["optimizer"] = "SGDOptimizerVar"
        flag_optimizer = 1
    else:
        try:
            if optimizer and flag_optimizer == 0:
                config_dict["optimizer"] = "AdamOptimizer"
                flag_optimizer = 1
        except NameError:
            print("Optimizer is None")
            pass

    config_dict["learning_rate"] = lr
    config_dict["batch_size"] = batch_size
    config_dict["epochs"] = epochs
    config_dict["use_early_stopping"] = use_early_stopping
    config_dict["feature_sizes"] = feature_sizes
    config_dict["dataset_names"] = dataset_names
    config_dict["start_time"] = current_time
    config_dict["weight_decay"] = weight_decay
    config_dict["DEBUG_MODE"] = DEBUG_MODE
    config_dict["optimizer"] = f"{optimizer.actual_optimizer_name}"
    config_dict["scheduler"] = f"{scheduler.actual_scheduler_name}"
    config_dict["saved weights"] = f"{save_dir}"
    # add dataset name
    
    # config_dict["amsgrad"] = amsgrad
    # config_dict["betas"] = betas
    # config_dict["eps"] = eps
    # config_dict["T_max"] = T_max
    # config_dict["eta_min"] = eta_min
    # config_dict["last_epoch"] = last_epoch
    # config_dict["num_workers"] = num_workers    
    config_dict["seed_value"] = seed_value
    # Create the directory path in a platform-independent way
    directory_path_conf = os.path.join(ROOT, "configs")

    # Check if the directory exists; if not, create it
    if not os.path.exists(directory_path_conf):
        os.makedirs(directory_path_conf)
    # Create a dynamic filename based on current time and learning rate
    filename = f"config_{current_time}_lr_{lr}_batch_{batch_size}_ep{epochs}.yaml"
    full_path = os.path.join(directory_path_conf, filename)

    # Write to the dynamically named config.yaml file
    with open(full_path, "w") as f:
        yaml.dump(config_dict, f)

    # TODO:
    # Add optimizer/scheduler/network decay rate to the config file

    # metrics_logger_retrieval.to(device)
    for feature_size in feature_sizes:

        model_cub200 = resnet50.ResNet50(feature_size, num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)
        fine_tuned_weights = torch.load(f'{save_dir}/resnet50_{feature_size}_CUB-200_batchsize_{batch_size}_lr_{lr}.pth')
        save_dir = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned/experiment_gpu_1/weights"
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

        # retrieval_metrics_cifar10 = evaluate_on_retrieval(model_cifar10, trainloader_cifar10, testloader_cifar10, batch_size, device=device)
        # retrieval_metrics_cub200 = evaluate_on_retrieval(model_cub200, trainloader_cub200, testloader_cub200, metrics_logger_retrieval, batch_size, device=device)
        retrieval_metrics_cub200 = evaluate_on_retrieval(model_cub200, trainloader_cub200, testloader_cub200, batch_size, device=device)
        
        
        print(f"\nRetrieval metrics for CUB-200: {retrieval_metrics_cub200}")
        metrics_logger_retrieval.log_retrieval_metrics(feature_size, retrieval_metrics_cub200, dataset_names[0])
        # print(f"\nRetrieval metrics for CUB-200 against feature size: {retrieval_metrics_cub200}")

        print(f"Retrieval evaluation for CUB-200 with feature size {feature_size} is complete!\n")
        retrieval_metrics_cub200[feature_size] = {'mAP': retrieval_metrics_cub200['mAP'], 'R@1': retrieval_metrics_cub200['R@1'], 
                                    'R@5': retrieval_metrics_cub200['R@5'], 'R@10': retrieval_metrics_cub200['R@10']}


if __name__ == "__main__":
    main_resnet()

    # def get_predictions(model, dataloader, device):
    #     num_samples = len(dataloader.dataset)
    #     # num_features = model.fc.in_features  # Assuming the model's fully connected layer is named 'fc'
    #     num_features = model.classifier.in_features  # Assuming the model's fully connected layer is named 'classifier'

    #     # Pre-allocate memory
    #     predictions = np.zeros((num_samples, model.num_classes))
    #     ground_truths = np.zeros(num_samples)        
    #     start_idx = 0        
    #     model.eval()

    #     with torch.inference_mode():
    #         for images, labels in dataloader:
    #             images, labels = images.to(device), labels.to(device)
    #             outputs = model(images)
                
    #             end_idx = start_idx + outputs.shape[0]
    #             predictions[start_idx:end_idx, :] = outputs.cpu().numpy()
    #             ground_truths[start_idx:end_idx] = labels.cpu().numpy()
                
    #             start_idx = end_idx  # Update the starting index for the next batch
        
    #     return predictions, ground_truths
    
    

