import torch
from config import config

def train(model, train_dataloader):

    model.train()
    train_loss = 0.0
    train_acc = 0.0

    batch = 1
    total_batches = len(train_dataloader)

    print("----Training CNN Model")
    for images, labels in train_dataloader:

        print(f"--------{batch} of {total_batches} batches.")
        batch += 1

        images, labels = images.to(config.device), labels.to(config.device)
        config.optimizer.zero_grad()
        outputs = model(images)

        if config.classification == "binary-classification":
            labels = labels.view(-1,1).float()
        losses = config.criterion(outputs, labels)
        losses.backward()
        config.optimizer.step()
        train_loss += losses.item() * images.size(0)

        if config.classification == "multi-classification":
            _, preds = torch.max(outputs, 1)
        else:
            preds = (torch.sigmoid(outputs) > 0.5).float()
        correct_tensor = preds.eq(labels.data.view_as(preds))
        accuracy = torch.mean(correct_tensor.float())
        train_acc += accuracy.item() * images.size(0)

    train_loss = train_loss / len(train_dataloader.dataset)
    train_acc  = train_acc / len(train_dataloader.dataset)


    print(f"--------Loss: {train_loss}")
    print(f"--------acc: {train_acc}")
    return train_loss, train_acc

def validate(model, val_dataloader):

    model.eval()
    val_loss = 0.0
    val_acc =0.0

    batch = 1
    total_batches = len(val_dataloader)

    print("----Validating CNN Model")
    with torch.no_grad():

        for images, labels in val_dataloader:

            print(f"--------{batch} of {total_batches} batches.")
            batch += 1

            images, labels = images.to(config.device), labels.to(config.device)
            outputs = model(images)

            if config.classification == "binary-classification":
                labels = labels.view(-1, 1).float()
            losses = config.criterion(outputs, labels)
            val_loss += losses.item() * images.size(0)

            if config.classification == "multi-classification":
                _, preds = torch.max(outputs, 1)
            else:
                preds = (torch.sigmoid(outputs) > 0.5).float()
            correct_tensor = preds.eq(labels.data.view_as(preds))
            accuracy = torch.mean(correct_tensor.float())
            val_acc += accuracy.item() * images.size(0)

        val_loss = val_loss / len(val_dataloader.dataset)
        val_acc = val_acc / len(val_dataloader.dataset)

        print(f"--------Loss: {val_loss}")
        print(f"--------acc: {val_acc}")
        return val_loss, val_acc


