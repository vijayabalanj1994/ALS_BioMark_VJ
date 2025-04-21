import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from process_data.load_data import read_files, split_dataset, ALS_Dataset
from process_data.augemt_data import augment_images
from process_data.utils import gray_compute_mean_std
from model.cnn import MultiCNNModel
from model.resnet18 import PretrainedResNet18, SE_PretrainedResNet18, CBAM_PretrainedResNet18
from model.densenet121 import PretrainedDenseNet121, SE_PretrainedDenseNet121
from model.utils import train_model, evaluate_model
from config import config

# reading the data files
images, labels = read_files()

#spliting to train, valid and test
train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels = split_dataset(images, labels)

# augmenting the train images
train_image_paths, train_labels = augment_images(train_image_paths, train_labels)

# transform function to Normalizing the images
mean, std = gray_compute_mean_std(train_image_paths + val_image_paths + test_image_paths)
transform = transforms.Compose([
    transforms.Resize((400,400)),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])

# converting dataset to torch Dataset objects
train_dataset = ALS_Dataset(train_image_paths, train_labels, transform=transform)
val_dataset = ALS_Dataset(val_image_paths, val_labels, transform=transform)
test_dataset = ALS_Dataset(test_image_paths, test_labels, transform=transform)

# the dataloaders
config.train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
config.val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
config.test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# CNN model
if config.model == "CNN":
    config.model = MultiCNNModel().to(config.device)
    config.criterion = nn.CrossEntropyLoss()
    config.optimizer = torch.optim.Adam(config.model.parameters(), lr=config.lr)
    config.scheduler = ReduceLROnPlateau(config.optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001)
    #train_model()
    print("Loading trained model....")
    config.model.load_state_dict(torch.load("C:\\Users\\vijay\\Neuro_BioMark\\ALS_BioMark_VJ\\saved_models\\model_weights.pth", weights_only=True))
    print(".... Loaded")
    evaluate_model()

# PretrainedResNet18
if config.model == "PretrainedResNet18":
    config.model = PretrainedResNet18().to(config.device)
    config.criterion = nn.CrossEntropyLoss()
    config.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, config.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    config.scheduler = ReduceLROnPlateau(config.optimizer, mode="min", factor=0.5, patience=5, min_lr=0.000001)
    train_model()
    evaluate_model()

# SE_PretrainedResNet18
if config.model == "SE_PretrainedResNet18":
    config.model = SE_PretrainedResNet18().to(config.device)
    config.criterion = nn.CrossEntropyLoss()
    config.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, config.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    config.scheduler = ReduceLROnPlateau(config.optimizer, mode="min", factor=0.5, patience=5, min_lr=0.000001)
    train_model()
    evaluate_model()

# CBAM_PretrainedResNet18
if config.model == "CBAM_PretrainedResNet18":
    config.model = CBAM_PretrainedResNet18().to(config.device)
    config.criterion = nn.CrossEntropyLoss()
    config.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, config.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    config.scheduler = ReduceLROnPlateau(config.optimizer, mode="min", factor=0.5, patience=5, min_lr=0.000001)
    train_model()
    evaluate_model()

if config.model == "PretrainedDenseNet121":
    config.model = PretrainedDenseNet121().to(config.device)
    config.criterion = nn.CrossEntropyLoss()
    config.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, config.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
    config.scheduler = ReduceLROnPlateau(config.optimizer, mode="min", factor=0.5, patience=5, min_lr=0.000001)
    train_model()
    evaluate_model()


config.model = SE_PretrainedDenseNet121().to(config.device)
config.criterion = nn.CrossEntropyLoss()
config.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, config.model.parameters()), lr=config.lr, weight_decay=config.weight_decay)
config.scheduler = ReduceLROnPlateau(config.optimizer, mode="min", factor=0.5, patience=5, min_lr=0.000001)
train_model()
evaluate_model()
