import os
import shutil
from process_data.load_data import read_files, split_dataset
from process_data.augemt_data import augment_images
import cv2

images, labels = read_files()
train_image_paths, val_image_paths, test_image_paths, train_labels, val_labels, test_labels = split_dataset(images, labels)

# Path to the 'augmented_train_data' folder
data_folder = os.path.join(os.path.dirname(__file__), 'dataset', 'augmented_train_data')
# Delete the folder and everything inside it
if os.path.exists(data_folder):
    shutil.rmtree(data_folder)
    print(f"Deleted folder: {data_folder}")
else:
    print(f"Folder does not exist: {data_folder}")

train_image_paths, train_labels = augment_images(train_image_paths, train_labels)
