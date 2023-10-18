import sys
sys.path.append('/home/alabutaleb/Desktop/myprojects/compression_retrieval_proj/mvp/src')
print(sys.path)

import os
from dataloaders.cifar10 import DataLoaderCIFAR10
from exp_logging.metricsLogger import MetricsLogger
from torchvision.models import resnet50, ResNet50_Weights

from dataloaders.cub200loader import DataLoaderCUB200
from utils.metrics import calculate_map, calculate_recall_at_k
import models.resnet50 as resnet50
# import models.resnet50_v2 as resnet50_v2
import torch
import torch.nn as nn
from optimizers.adam import AdamOptimizer
from trainers.resnets_trainer import ResnetTrainer
from loss.ce import CustomCrossEntropyLoss
import matplotlib.pyplot as plt
from utils.gallery import extract_gallery_features
from utils.similarities import image_retrieval, evaluate_on_retrieval

import numpy as np









def main_resnet():

    def get_predictions(model, dataloader, device):
        num_samples = len(dataloader.dataset)
        # num_features = model.fc.in_features  # Assuming the model's fully connected layer is named 'fc'
        num_features = model.classifier.in_features  # Assuming the model's fully connected layer is named 'classifier'

        # Pre-allocate memory
        predictions = np.zeros((num_samples, model.num_classes))
        ground_truths = np.zeros(num_samples)        
        start_idx = 0        
        model.eval()

        with torch.inference_mode():
            for images, labels in dataloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                
                end_idx = start_idx + outputs.shape[0]
                predictions[start_idx:end_idx, :] = outputs.cpu().numpy()
                ground_truths[start_idx:end_idx] = labels.cpu().numpy()
                
                start_idx = end_idx  # Update the starting index for the next batch
        
        return predictions, ground_truths
    
    def ensure_directory_exists(directory_path):
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            print(f"Directory {directory_path} created.")
        else:
            print(f"Directory {directory_path} already exists.")

  
    def create_optimizer(model, lr=0.001):        
        optimizer = AdamOptimizer(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        return optimizer.get_optimizer()


    def train_and_evaluate(dataset_name, model, optimizer, 
                           criterion, trainloader, testloader, 
                           use_early_stopping, device, num_classes, metrics_logger,
                           epochs, feature_size, save_dir):
        
        print(f"Training with {dataset_name} dataset and feature size: {feature_size}")
        trainer = ResnetTrainer(model, optimizer, criterion, trainloader, 
                                testloader, feature_size, use_early_stopping,
                                device, num_classes, metrics_logger, epochs=epochs, 
                                dataset_name=dataset_name)
        
        model, training_loss, training_accuracy = trainer.train_model()
        average_validation_loss, validation_accuracy_percentage, batch_losses = trainer.evaluate_model()

        torch.save(model.state_dict(), f'{save_dir}/resnet50_{feature_size}_{dataset_name}_fine_tuned.pth')
        print(f"Model saved at {save_dir}/resnet50_{feature_size}_{dataset_name}_fine_tuned.pth")
        return model, training_loss, average_validation_loss, validation_accuracy_percentage

    
    #### set device #####
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("#"*30)
    print(f"You are curently using {device} device")
    print("#"*30)

    #### directory to save fine tuned weights #####
    save_dir = "/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/resnet50_finetuned"
    data_root ="/media/alabutaleb/09d46f11-3ed1-40ce-9868-932a0133f8bb/data/cub200/"
    # Create dictionaries to store metrics
    metrics_cifar10 = {}
    metrics_cub200 = {}


    ensure_directory_exists(save_dir)
           
    #### hyperparameters #####
    batch_size  = 32
    epochs = 1
    use_early_stopping=False
    print("/\\"*30)
    print(f"You are using batch size: {batch_size}")
    print(f"You are using epochs: {epochs}")
    print(f"You are using early stopping: {use_early_stopping}")
    print("/\\"*30)
    #### get data #####root, batch_size=32,num_workers=10

    dataloadercifar10 = DataLoaderCIFAR10(batch_size=batch_size, data_root='./data')
    trainloader_cifar10, testloader_cifar10, num_classes_cifar10 = dataloadercifar10.get_dataloaders()
    dataloadercub200 = DataLoaderCUB200(data_root, batch_size=batch_size, num_workers=10)
    trainloader_cub200, testloader_cub200 = dataloadercub200.get_dataloaders()

    num_classes_cub200 = dataloadercub200.get_number_of_classes()
    #### prep logger ####

    # Initializing metrics logger for later use in logging metrics
    exit(f"code below won't work!")
    dataset_names=["cifar10", "cub200"]
    metrics_logger = MetricsLogger(dataset_names)


    #### get loss for fine tuning #####
    criterion = CustomCrossEntropyLoss()

    #### feature sizes #####
    # feature_sizes = [2048, 1024, 512, 256, 128, 64, 32, 16]
    feature_sizes = [16]

    
    #### train model #####
    # Main loop to iterate over each feature size
    for feature_size in feature_sizes:                        
        # #### get model ##### 
        # model = resnet50_v2.DynamicResNet50(feature_size, num_classes_cifar10, weights=ResNet50_Weights.DEFAULT)
        
        model = resnet50.ResNet50(feature_size, num_classes_cifar10, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)
        model.fine_tune_mode()
        model = model.to(device)      
        optimizer  = create_optimizer(model)
        # Training CIFAR-10
        model_cifar10, train_loss_cifar10, val_loss_cifar10, val_acc_cifar10= train_and_evaluate("CIFAR-10", model, optimizer, 
                                                                       criterion, trainloader_cifar10, 
                                                                       testloader_cifar10,use_early_stopping,
                                                                       device, num_classes_cifar10,metrics_logger,
                                                                       epochs, feature_size, save_dir)
        metrics_cifar10[feature_size] = {'train_loss': train_loss_cifar10, 'val_loss': val_loss_cifar10, 'val_acc': val_acc_cifar10}

    
        #### get fresh things for cub200 #####
        # model = resnet50_v2.DynamicResNet50(feature_size, num_classes_cifar10, weights=ResNet50_Weights.DEFAULT)
        model = resnet50.ResNet50(feature_size, num_classes_cub200, weights=ResNet50_Weights.DEFAULT, pretrained_weights=None)
        model.fine_tune_mode()
        model = model.to(device)
        optimizer  = create_optimizer(model)
        # Training CUB-200
        model_cub200, train_loss_cub200, val_loss_cub200, val_acc_cub200 = train_and_evaluate("CUB-200", model, optimizer, 
                                                                      criterion, trainloader_cub200, 
                                                                      testloader_cub200, use_early_stopping,
                                                                      device, num_classes_cub200,
                                                                      epochs, feature_size, save_dir)
        metrics_cub200[feature_size] = {'train_loss': train_loss_cub200, 'val_loss': val_loss_cub200, 'val_acc': val_acc_cub200}
        
        # Plot and log classification performance
        # models_metrics = {
        #         'cifar10': {'loss': loss_value, 'mAP': mAP_value, 'R@1': R1_value, 'R@5': R5_value, 'R@10': R10_value},
        #         'cub200': {'loss': loss_value, 'mAP': mAP_value, 'R@1': R1_value, 'R@5': R5_value, 'R@10': R10_value}
        #             }
        # models_metrics = {
        #     'cifar10': {'loss': train_loss_cifar10[-1], 'mAP': val_acc_cifar10[-1]},
        #     'cub200': {'loss': train_loss_cub200[-1], 'mAP': val_acc_cub200[-1]}
        #             }

        # MetricsLogger.plot_classification_performance(models_metrics['cifar10'], models_metrics['cub200'], feature_size)
        

        

        # After training, calculate the metrics (mAP, R@1, R@5, R@10) for each dataset

        # Loop to calculate metrics for each dataset after training
        # for dataset_name, model, trainloader, training_loss in [("cub200", model_cub200, train_loss_cub200, train_loss_cub200)]:
        # for dataset_name, model, testloader, last_training_loss in [("cifar10", model_cifar10, testloader_cifar10, train_loss_cifar10), 
        #                                             ("cub200", model_cub200, testloader_cub200, train_loss_cub200)]:
        #     metrics_logger.log_and_calculate_metrics(dataset_name, feature_size, model, testloader, last_training_loss, device)



    # After all experiments, plot the metrics using the metrics logger
    # metrics_logger.plot_metrics()

#consider moving all of the code below to a separate file
    for feature_size in feature_sizes:
        # Load the previously saved models for CIFAR-10 and CUB-200
        # save dir is the directory where the fine tuned weights are saved
        model_cifar10 = resnet50.ResNet50(feature_size, num_classes_cifar10, weights='IMAGENET1K_V1', pretrained_weights=None)
        fine_tuned_weights = torch.load(f'{save_dir}/resnet50_{feature_size}_CIFAR-10_fine_tuned.pth')    
        model_cifar10.load_state_dict(fine_tuned_weights)
        model_cifar10.feature_extractor_mode()
        model_cifar10 = model_cifar10.to(device)


        model_cub200 = resnet50.ResNet50(feature_size, num_classes_cub200, weights='IMAGENET1K_V1', pretrained_weights=None)
        fine_tuned_weights = torch.load(f'{save_dir}/resnet50_{feature_size}_CUB-200_fine_tuned.pth')
        model_cub200.load_state_dict(fine_tuned_weights)
        model_cub200.feature_extractor_mode()
        model_cub200 = model_cub200.to(device)
        
        
        # Test and evaluate models on image retrieval task
        # print(model_cifar10)
        # exit(model_cub200)

     
        retrieval_metrics_cifar10 = evaluate_on_retrieval(model_cifar10, trainloader_cifar10, testloader_cifar10, batch_size, device=device)
        retrieval_metrics_cub200 = evaluate_on_retrieval(model_cub200, trainloader_cub200, testloader_cub200, batch_size, device=device)
        
        # Log and plot retrieval performance
        # metrics_logger.log_and_plot_retrieval_performance(retrieval_metrics_cifar10, retrieval_metrics_cub200, feature_size)
        # Log and plot metrics using the MetricsLogger
        # metrics_logger.log_and_plot_classification_metrics(models_metrics, feature_size)



if __name__ == "__main__":

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
    
    main_resnet()

