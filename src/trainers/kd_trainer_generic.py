
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchmetrics.classification import Accuracy
import torch.nn.functional as F

class GenericKDLossTrainer:
    def __init__(self, teacher_model, student_model, strategy, criterion, optimizer, num_classes, device, use_early_stopping, ta_model=None, scheduler=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.teacher_model = teacher_model.to(self.device)
        self.student_model = student_model.to(self.device)
        self.ta_model = ta_model.to(self.device) if ta_model else None
        self.strategy = strategy
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_classes = num_classes
        # ... (other initializations)

    def distillation_loss(self, student_inputs, student_outputs, labels, alpha=0.5):
        if self.strategy == "vanilla_kd":
            with torch.inference_mode():
                teacher_outputs = self.teacher_model(student_inputs)
            soft_labels = F.softmax(teacher_outputs / self.temperature, dim=1)
        elif self.strategy == "TAKD_kd":
            with torch.inference_mode():
                ta_outputs = self.ta_model(student_inputs)
            soft_labels = F.softmax(ta_outputs / self.temperature, dim=1)
        elif self.strategy == "srd":    # Semantic Representational Distillation
            with torch.inference_mode():
                exit(f"You have not coded SRD yet")
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        distillation_loss = F.kl_div(student_soft, soft_labels, reduction='batchmean')
        student_loss = self.criterion(student_outputs, labels)
        return alpha * distillation_loss + (1 - alpha) * student_loss

    # ... (train_epoch and evaluate methods, similar to previous classes)
    # Modify the train() method to use distillation_loss
    def train(self, train_loader, val_loader, num_epochs):
        # ... (Implementation of the training loop)
        return self.student_model

    # ... (plot_loss_vs_epoch and plot_acc_vs_epoch methods)
