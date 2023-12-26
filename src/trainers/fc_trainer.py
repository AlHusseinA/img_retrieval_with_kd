import torch
from torch import nn
from utils.helpers_for_pca_exp import batch_features


def fc_trainer(new_fc, epochs, optimizer, criterion, train_loader, test_loader, train_features, test_features, device):
    new_fc.to(device)
    batch_size = train_loader.batch_size  # Assuming batch size is same for train and test loaders
    # train_feature_batches = batch_features(train_features, batch_size)
    # test_feature_batches = batch_features(test_features, batch_size)
    train_feature_batches = batch_features(train_features, train_loader.batch_size)
    test_feature_batches = batch_features(test_features, test_loader.batch_size)

    for epoch in range(epochs):
        new_fc.train()
        train_loss, train_correct, total_train = 0, 0, 0
        # Iterate over both dataloader and features
        # for (_, labels), features in zip(train_loader, train_feature_batces):
        for (_, labels), feature_batch in zip(train_loader, train_feature_batches):
            # features, labels = features.to(device), labels.to(device)
            # features = torch.from_numpy(features).to(device).float()  # Convert to tensor and send to device
            # labels = labels.to(device)

            # Adjust the feature batch size if it doesn't match the labels batch size
            feature_batch = feature_batch[:len(labels)]
            features = torch.from_numpy(feature_batch).to(device).float()
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = new_fc(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            train_correct += (predicted == labels).sum().item()

        new_fc.eval()
        val_loss, val_correct, total_val = 0, 0, 0


        with torch.no_grad():
            for (_, labels), feature_batch in zip(test_loader, test_feature_batches):
                # Adjust the feature batch size to match the labels batch size
                feature_batch = feature_batch[:len(labels)]
                features = torch.from_numpy(feature_batch).to(device).float()
                labels = labels.to(device)

                outputs = new_fc(features)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                val_correct += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(test_loader.dataset)
        train_accuracy = 100 * train_correct / total_train
        val_accuracy = 100 * val_correct / total_val

        print(f"Epoch: {epoch+1}/{epochs} | "
              f"Training Loss: {avg_train_loss:.4f} | "
              f"Validation Loss: {avg_val_loss:.4f} | "
              f"Training Accuracy: {train_accuracy:.2f}% | "
              f"Validation Accuracy: {val_accuracy:.2f}%")

    return new_fc
