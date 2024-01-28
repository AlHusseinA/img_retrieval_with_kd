# import torch
# import torch.nn as nn
# import torch.nn.functional as F


# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T

#     def forward(self, y_s, y_t, class_weights=None):
#         p_s = F.log_softmax(y_s/self.T, dim=1)
#         p_t = F.softmax(y_t/self.T, dim=1)
#         if class_weights is not None:
#             _, preds_t = torch.max(y_t, 1)
#             weights = class_weights[preds_t]
#             weights = weights.unsqueeze(1)
#             l_kl = F.kl_div(p_s, p_t, reduction='none')
#             loss = torch.sum(l_kl * weights) * (self.T**2) / y_s.shape[0]
#         else:
#             loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]
#         return loss
    


import torch
import torch.nn as nn
import torch.nn.functional as F

# modified from from SRD paper


class DistillKL(nn.Module):
    """Distilling the Knowledge in a Neural Network"""
    def __init__(self, criterion, T, alpha=0.5):
        super(DistillKL, self).__init__()
        self.T = T
        self.alpha = alpha
        self.criterion = criterion

    def forward(self, y_s, y_t, labels, class_weights=None):
        p_s = F.log_softmax(y_s/self.T, dim=1)
        p_t = F.softmax(y_t/self.T, dim=1)

        if class_weights is not None:
            _, preds_t = torch.max(y_t, 1)
            weights = class_weights[preds_t]
            weights = weights.unsqueeze(1)
            l_kl = F.kl_div(p_s, p_t, reduction='none')
            distillation_loss = torch.sum(l_kl * weights) * (self.T**2) / y_s.shape[0]
        else:
            distillation_loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]

        student_loss = self.criterion(y_s, labels)
        total_loss = self.alpha * distillation_loss + (1 - self.alpha) * student_loss
        return total_loss



# class DistillKL(nn.Module):
#     """Distilling the Knowledge in a Neural Network"""
#     def __init__(self, T):
#         super(DistillKL, self).__init__()
#         self.T = T

#     def forward(self, y_s, y_t, class_weights=None):
#         p_s = F.log_softmax(y_s/self.T, dim=1)
#         p_t = F.softmax(y_t/self.T, dim=1)

#         if class_weights is not None:
#             # Check if class_weights and y_t are on the same device
#             assert class_weights.device == y_t.device, "class_weights and y_t should be on the same device"

#             _, preds_t = torch.max(y_t, 1)
            
#             # Debugging: Check the range of preds_t
#             assert preds_t.min() >= 0 and preds_t.max() < len(class_weights), \
#                    "Values in preds_t are out of bounds for class_weights"

#             weights = class_weights[preds_t]
#             weights = weights.unsqueeze(1)

#             l_kl = F.kl_div(p_s, p_t, reduction='none')
#             loss = torch.sum(l_kl * weights) * (self.T**2) / y_s.shape[0]
#         else:
#             loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T**2) / y_s.shape[0]

#         # Additional Debugging: Check for NaN or Inf in loss
#         if torch.isnan(loss) or torch.isinf(loss):
#             print("Warning: NaN or Inf in loss")
#             print("p_s:", p_s)
#             print("p_t:", p_t)
#             print("y_s:", y_s)
#             print("y_t:", y_t)
#             print("weights:", weights if class_weights is not None else "N/A")

#         return loss
