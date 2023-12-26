import torch
from loss.ce import CustomCrossEntropyLoss



def pca_transform(features, pca):
    return torch.from_numpy(pca.transform(features.cpu())).to(features.device)


def train_eval_fc_pca(model, train_loader, test_loader, pca, optimizer, device):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = CustomCrossEntropyLoss()
    # model.train()  # Set the model to training mode
    
    train_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        with torch.inference_mode():
            features = model(inputs)
            compressed_features = pca_transform(features, pca)

        outputs = model.fc(compressed_features)
        loss = criterion(outputs, labels)
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        loss.backward()
        optimizer.step()

    train_accuracy = 100 * correct / total if total > 0 else 0

    validation_accuracy = evaluate_compression(model, test_loader, pca, device)

    print(f'Training Loss: {train_loss / len(train_loader)}, '
          f'Training Accuracy: {train_accuracy}%, '
          f'Validation Accuracy: {validation_accuracy}%')

    return model


def evaluate_compression(model, test_loader, pca, device="cuda"):
    model.eval()
    accuracy = 0
    total = 0
    validation_loss = 0.0
    criterion = CustomCrossEntropyLoss()

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        with torch.inference_mode():
            features = model(inputs)
            compressed_features = pca_transform(features, pca)
            outputs = model.fc(compressed_features)
            loss = criterion(outputs, labels)
            validation_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()

    accuracy = 100 * accuracy / total if total > 0 else 0
    validation_loss = validation_loss / len(test_loader) if len(test_loader) > 0 else 0

    print(f'Validation Loss: {validation_loss}, Validation Accuracy: {accuracy}%')

    return accuracy




