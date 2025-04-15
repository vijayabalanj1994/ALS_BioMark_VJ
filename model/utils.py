import os
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, matthews_corrcoef, confusion_matrix
from config import config

def train(model, train_dataloader):

    model.train()
    train_loss = 0.0
    train_acc = 0.0

    batch = 1
    total_batches = len(train_dataloader)

    #print("--Training CNN Model")
    for images, labels in train_dataloader:

        #print(f"--------{batch} of {total_batches} batches.")
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


    #print(f"--------Loss: {train_loss}")
    #print(f"--------acc: {train_acc}")
    return train_loss, train_acc

def validate(model, val_dataloader):

    model.eval()
    val_loss = 0.0
    val_acc =0.0

    batch = 1
    total_batches = len(val_dataloader)

    #print("--Validating CNN Model")
    with torch.no_grad():

        for images, labels in val_dataloader:

            #print(f"--------{batch} of {total_batches} batches.")
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

        #print(f"--------Loss: {val_loss}")
        #print(f"--------acc: {val_acc}")
        return val_loss, val_acc

def train_model():

    epoch_train_loss = []
    epoch_train_acc = []
    epoch_val_loss =[]
    epoch_val_acc = []

    print("Training Model----")
    for epo in range(config.epoch):
        print(f"----{epo+1} of {config.epoch} epoch:-")
        # Training
        train_loss, train_acc = train(config.model, config.train_loader)
        epoch_train_loss.append(train_loss)
        epoch_train_acc.append(train_acc)

        # Validation
        val_loss, val_acc = validate(config.model, config.val_loader)
        epoch_val_loss.append(val_loss)
        epoch_val_acc.append(val_acc)

        #scheduler step
        config.scheduler.step(val_loss)

        print(f"-------- train_loss: {train_loss:.2f} train_acc: {train_acc:.2f} val_loss: {val_loss:.2f} val_acc: {val_acc:.2f}")

    # path to store training results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True) # will create results directory if not exists
    file_path = os.path.join(results_dir, 'training_results.csv')

    # storing the raining results
    training_results = pd.DataFrame({
        "epoch": list(range(1,config.epoch+1)),
        "train_loss": epoch_train_loss,
        "train_acc": epoch_train_acc,
        "val_loss": epoch_val_loss,
        "val_acc": epoch_val_acc
    })
    training_results.to_csv(file_path, index=False)

    # path to store the trained model
    saved_model_dir = os.path.join(os.path.dirname(__file__), '..', 'saved_models')
    os.makedirs(saved_model_dir, exist_ok=True)  # will create results directory if not exists
    model_path = os.path.join(saved_model_dir, 'model_weights.pth')
    torch.save(config.model.state_dict(), model_path)

    print("----Done Training")

def sensitivity_specificity(y_true, y_pred, labels=None):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    num_classes = cm.shape[0]
    sensitivities = []
    specificities = []

    for i in range(num_classes):
        TP = cm[i, i]
        FN = np.sum(cm[i, :]) - TP
        FP = np.sum(cm[:, i]) - TP
        TN = np.sum(cm) - (TP + FN + FP)

        sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        sensitivities.append(sensitivity)
        specificities.append(specificity)

    return sensitivities, specificities

def evaluate_model():

    print("\nEvaluating Model----")
    config.model.eval()
    test_loss = 0.0
    text_acc = 0.0
    text_error = 0
    y_true = []
    y_pred = []
    labels = None

    with torch.no_grad():
        for images, labels in config.test_loader:
            images, labels = images.to(config.device), labels.to(config.device)
            outputs = config.model(images)

            if config.classification == "multi-classification":
                _, preds = torch.max(outputs, 1)
            else:
                preds = (torch.sigmoid(outputs) > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if config.classification == "multi-classification":
        labels = [0,1,2]
    else:
        labels = [0,1]

    acc = accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    sens, spec = sensitivity_specificity(y_true, y_pred, labels=labels)

    # Create a DataFrame for saving
    results_df = pd.DataFrame({
        'Class': labels,
        'Sensitivity': sens,
        'Specificity': spec
    })

    # Add overall metrics
    overall_metrics = pd.DataFrame({
        'Class': ['Overall'],
        'Sensitivity': [np.nan],
        'Specificity': [np.nan],
        'Accuracy': [acc],
        'MCC': [mcc]
    })

    # Combine and save
    results_df = pd.concat([results_df, overall_metrics], ignore_index=True)
    # path to store training results
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    os.makedirs(results_dir, exist_ok=True)  # will create results directory if not exists
    file_path = os.path.join(results_dir, 'evaluation_results.csv')
    results_df.to_csv(file_path, index=False)

    print("----Saved Evaluation Results")