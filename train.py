import torch
import copy
from torch.utils.data import DataLoader


def train(num_epochs, batch_size, criterion, optimizer, model, train_dataset, val_dataset):
    train_error = []
    val_error = []
    train_acc = []
    val_acc = []
    best_acc = 0
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    for epoch in range(num_epochs):
        model.train()
        epoch_average_loss = 0.0
        epoch_val_loss = 0.0
        epoch_train_correct = 0.0
        epoch_val_correct = 0.0

        for (X_batch, y_real, imagename) in train_loader:
            y_pre = model(X_batch.to('cuda'))
            loss = criterion(y_pre, y_real.to('cuda'))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, predicted = torch.max(y_pre.data, 1)
            epoch_train_correct += (predicted == y_real.to('cuda')).sum()
            epoch_average_loss += loss.item() * batch_size / len(train_dataset)
        epoch_train_acc = epoch_train_correct / len(train_dataset)
        train_error.append(epoch_average_loss)
        train_acc.append(epoch_train_acc)

        model.eval()
        with torch.no_grad():
            for (X_batch, y_real, imagename) in val_loader:
                y_pre = model(X_batch.to('cuda'))
                loss = criterion(y_pre, y_real.to('cuda'))
                optimizer.zero_grad()
                _, predicted = torch.max(y_pre.data, 1)
                epoch_val_correct += (predicted == y_real.to('cuda')).sum()
                epoch_val_loss += loss.item() * batch_size / len(val_dataset)
        epoch_val_acc = epoch_val_correct / len(val_dataset)
        val_error.append(epoch_val_loss)
        val_acc.append(epoch_val_acc)
        print('Epoch [{}/{}], Train Loss: {:.4f}, Val Loss: {:.4f} , Train accuracy: {:.4f}, Val accuracy: {:.4f}'
              .format(epoch + 1, num_epochs, epoch_average_loss, epoch_val_loss, epoch_train_acc, epoch_val_acc))
        if best_acc < epoch_val_acc:
            best_acc = epoch_val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    return train_error, val_error, train_acc, val_acc