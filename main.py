import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from process_data.load_data import read_files, split_dataset, ALS_Dataset
from process_data.augemt_data import augment_images
from process_data.utils import gray_compute_mean_std
from model.cnn import MultiCNNModel
from model.utils import train, validate
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
train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

# CNN model
model = MultiCNNModel().to(config.device)
config.criterion = nn.CrossEntropyLoss()
config.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
scheduler = ReduceLROnPlateau(config.optimizer, mode='min', factor=0.5, patience=5, min_lr=0.00001)

train(model, train_loader)
validate(model, val_loader)