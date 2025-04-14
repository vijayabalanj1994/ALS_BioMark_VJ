import os
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from config import config


def read_files():
    # reading the Image Key exel file
    label_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'RoI Image Key.xlsx')
    df = pd.read_excel(label_path, usecols=[0, 3], skiprows=1, engine="openpyxl")

    # dictionary to associate unique ID of images to images path in file
    images ={}
    # dictionary to associate unique ID of images to image class (Control, Concordant, Discordant)
    labels = {}

    # The path to image directory
    image_directory_path = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'AptamerROIs020623')

    for index, row in df.iterrows():
        #print("Index:- ", index, "\tImage No:- ", row["Image No"], "\tCategory:- ", row["Category"])
        images[row["Image No"]] = os.path.join(image_directory_path,f"{row['Image No']}.tif")

        if config.classification == "multi-classification":
            if row["Category"] == "Control":
                labels[row["Image No"]] = 0
            elif row["Category"] == "Concordant":
                labels[row["Image No"]] = 1
            elif row["Category"] == "Discordant":
                labels[row["Image No"]] = 2
            else:
                raise ValueError(f"Unexpected class label when reading file in prep_data.py:- {row['Category']}")
        elif config.classification == "binary-classification":
            if row["Category"] == "Control":
                labels[row["Image No"]] = 0
            elif row["Category"] == "Concordant" or row["Category"] == "Discordant":
                labels[row["Image No"]] = 1
            else:
                raise ValueError(f"Unexpected class label when reading file in prep_data.py:- {row['Category']}")

    return images, labels


def split_dataset(images, labels):

    if config.train_ratio + config.test_ratio + config.val_ratio != 1:
        raise ValueError(f" Train + Test + Val ratio = {config.train_ratio + config.test_ratio + config.val_ratio} and must be = 1")

    # splitting the dataset into train and (test+val) data
    train_image_paths, test_and_val_image_paths, train_labels, test_and_val_labels = train_test_split(
        list(images.values()), list(labels.values()), test_size=(1-config.train_ratio), random_state=config.seed, stratify=list(labels.values())
    )

    # computing split ratios to split (test+val) dataset into test and val data
    val_split = config.val_ratio/(config.val_ratio+config.test_ratio)
    test_split = 1- val_split

    # splitting the (test+val) dataset into train and val data
    val_image_paths, test_image_paths, val_labels, test_labels = train_test_split(
        test_and_val_image_paths, test_and_val_labels, test_size=test_split, random_state=config.seed, stratify=test_and_val_labels
    )

    return train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels


class ALS_Dataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        label = self.labels[idx]
        if self.tensform:
            image = self.transform(image)
        return image, label
