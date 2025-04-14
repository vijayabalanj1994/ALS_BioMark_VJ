import cv2
import numpy as np

def gray_compute_mean_std(image_paths):

    per_image_mean =[]
    per_image_std = []

    for image_path in image_paths:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        image = image /255.0
        per_image_mean.append(np.mean(image))
        per_image_std.append(np.std(image))

    mean = np.mean(per_image_mean)
    std = np.mean(per_image_std)

    return [mean], [std]