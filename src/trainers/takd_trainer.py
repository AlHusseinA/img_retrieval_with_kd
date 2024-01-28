import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Accuracy


class TAKDTrainer:
    def __init__(self, ta_model, student_model, criterion, optimizer, num_classes, device, use_early_stopping, scheduler=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ta_model = ta_model.to(self.device)  # Teacher Assistant model
        self.student_model = student_model.to(self.device)
        self.ta_model.eval()  # TA model is always in evaluation mode
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_classes = num_classes
        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.scheduler = scheduler
        self.min_epochs_for_early_stopping = 20
        self.actual_epochs_run = 0
        self.use_early_stopping = use_early_stopping
        self.metric_train = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        self.metric_val = Accuracy(task="multiclass", num_classes=self.num_classes).to(self.device)
        
        if self.use_early_stopping:
            self.max_val_accuracy = 0.0
            self.patience = 20
            self.counter = 0

        self.temperature = 3  # Temperature for softmax in distillation

    def distillation_loss(self, student_outputs, ta_outputs, labels, alpha=0.5):
        soft_labels = F.softmax(ta_outputs / self.temperature, dim=1)
        student_soft = F.log_softmax(student_outputs / self.temperature, dim=1)
        distillation_loss = F.kl_div(student_soft, soft_labels, reduction='batchmean')
        student_loss = self.criterion(student_outputs, labels)
        return alpha * distillation_loss + (1 - alpha) * student_loss

    # ... (train_epoch and evaluate methods, similar to previous class)
    # Modify the train() method for TAKD
    def train(self, train_loader, val_loader, num_epochs):
        # ... (Similar structure as previous class, but use ta_model instead of teacher_model)

        return self.student_model

    # ... (plot_loss_vs_epoch and plot_acc_vs_epoch methods)
