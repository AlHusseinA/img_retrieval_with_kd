import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics.classification import Accuracy
from utils.helper_functions import check_size
from exp_logging.metricsLogger_kd import MetricsLoggerKD

import torch.nn.functional as F

from sklearn.metrics import accuracy_score
class KnowledgeDistillationTrainer():
    def __init__(self, teacher_model, student_model, criterion, distill_loss, optimizer, scheduler, logger, num_classes, lr, device, log_save_folder_kd, use_early_stopping, T):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.teacher_model = teacher_model.to(self.device)  # this is vanilla unmodified resnet50
        self.student_model = student_model.to(self.device)  # this is resnet50 with final features layer compressed
        # self.ta_model = ta_model.to(self.device) if ta_model else None
        self.teacher_model.eval()  # Teacher model is always in evaluation mode
        self.criterion = criterion
        self.distill_loss = distill_loss
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.lr = lr
        self.logger = logger
        self.num_classes = num_classes
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.min_epochs_for_early_stopping = 20
        self.actual_epochs_run = 0
        self.use_early_stopping = use_early_stopping
        self.metric_train = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        self.metric_val = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        self.log_save_folder_kd = log_save_folder_kd

        self.student_model_size = check_size(self.student_model)
        self.teacher_model_size = check_size(self.teacher_model)
        if self.use_early_stopping:
            self.max_val_accuracy = 0.0
            self.patience = 20
            self.counter = 0

        # Distillation specific attributes
        self.T = T  # Temperature for softmax in distillation

    def check_early_stopping(self, avg_val_accuracy, counter, patience):
        if self.actual_epochs_run < self.min_epochs_for_early_stopping:
            return False  # Continue training
        
        if avg_val_accuracy > self.max_val_accuracy:
            self.max_val_accuracy = avg_val_accuracy
            self.counter = 0  # Reset counter
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch: {self.actual_epochs_run}.")

                # insert logic for config file

                return True  # Stop training
        return False  # Continue training
    
    # def distillation_loss(self, student_outputs, teacher_outputs, labels, alpha=0.5):
    #     soft_labels = F.softmax(teacher_outputs / self.temperature, dim=1)
    #     student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
    #     distillation_loss = F.kl_div(student_soft, soft_labels, reduction='batchmean')
    #     student_loss = self.criterion(student_outputs, labels)
    #     return alpha * distillation_loss + (1 - alpha) * student_loss

    def train_epoch(self, train_loader, distill_loss):
        self.student_model.train()
        self.teacher_model.eval()
        train_loss = 0.0

        # labels_list = []
        # student_outputs_list = []

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            self.optimizer.zero_grad()

            student_outputs = self.student_model(inputs)
            with torch.inference_mode():
                teacher_outputs = self.teacher_model(inputs)

            # using the DistillKL object for loss calculation
            loss = distill_loss(student_outputs, teacher_outputs, labels)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            # self.metric_train(student_outputs.softmax(dim=-1), labels)
            self.metric_train(student_outputs.softmax(dim=-1), labels)

            # ### Tony's idea
            # labels_list.append(torch.argmax(labels,dim=-1))
            # student_outputs_list.append(torch.argmax(student_outputs,dim=-1)))
            # ###

        # print("Accuracy ",accuracy_score(labels_list, student_outputs_list))
                                        
        if self.scheduler is not None:
            self.scheduler.step()

        train_acc = self.metric_train.compute().item() * 100
        self.metric_train.reset()

        return train_loss / len(train_loader), train_acc

    def evaluate(self, val_loader):
        self.student_model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.student_model(inputs)
                loss = self.criterion(outputs, labels)
                val_loss += loss.item()
                self.metric_val(outputs.softmax(dim=-1), labels)

        val_acc = self.metric_val.compute().item() * 100
        self.metric_val.reset()

        return val_loss / len(val_loader), val_acc
    
    def train(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch(train_loader, self.distill_loss)
            val_loss, val_acc = self.evaluate(val_loader)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            self.actual_epochs_run += 1

            if self.use_early_stopping:
                stop_training = self.check_early_stopping(val_acc, self.counter, self.patience)
                if stop_training:
                    break

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')

            if self.logger is not None:
                try:
                    self.logger.log_metrics(
                        student_size=self.student_model_size,  # Replace with actual student model size or relevant metric
                        epoch=self.actual_epochs_run,
                        training_loss=train_loss,
                        train_acc=train_acc,
                        avg_val_loss=val_loss,
                        val_acc=val_acc,
                        dataset_name="CUB200",  # Replace with your actual dataset name
                        log_save_path=self.log_save_folder_kd,  # Replace with your actual log save path
                        lr=self.lr, # Assuming this is how you access the learning rate
                        T=self.T
                    )
                except Exception as e:
                    print(f"An unexpected error occurred while logging metrics: {e}")
                    import traceback
                    print("Stack Trace:")
                    print(traceback.format_exc())

            self.metric_train.reset()  # Reset the training metric for the next epoch

        return self.student_model


# Usage:
# teacher_model = ...  # Load teacher model
# student_model = resnet34(pretrained=True)
# trainer = KnowledgeDistillationTrainer(...)
# trainer.train(train_loader, val_loader, num_epochs)
    







# def main():
#     teacher_model = load_teacher_model('path_to_resnet50_weights.pth')
#     student_model = resnet34(pretrained=True)

#     trainer = KnowledgeDistillationTrainer(teacher_model, student_model, ...)
#     trainer.run_training_loop(num_epochs=...)


