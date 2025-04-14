import os
import cv2
import numpy as np
#from PIL.ImageOps import grayscale
#from cv2.gapi import cmpNE

def augment_images(img_paths, labels):

    output_folder = 'dataset\\augmented_train_data'
    train_image_paths = []
    train_labels = []

    for img_path, label in zip(img_paths, labels):

        # the original image
        img_color = cv2.imread(img_path)
        img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        original_folder = os.path.join(output_folder, "original")
        if not os.path.exists(original_folder):
            os.makedirs(original_folder)
        output_path = os.path.join(original_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, img_color)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # flipping the image
        flipped_img = cv2.flip(img_color, 1)

        flip_folder = os.path.join(output_folder, "flipped")
        if not os.path.exists(flip_folder):
            os.makedirs(flip_folder)
        output_path = os.path.join(flip_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, flipped_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # rotating the image
        h, w = img_color.shape[:2]
        center = (w //2, h//2)
        m= cv2.getRotationMatrix2D(center, 90, 1)
        rotated_img = cv2.warpAffine(img_color, m, (h, w))

        rotate_folder = os.path.join(output_folder, "rotated")
        if not os.path.exists(rotate_folder):
            os.makedirs(rotate_folder)
        output_path = os.path.join(rotate_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, rotated_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # converting into grayscale (but retains 3-channels)
        grayscale_3channel_img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)

        gray_folder = os.path.join(output_folder, "gray")
        if not os.path.exists(gray_folder):
            os.makedirs(gray_folder)
        output_path = os.path.join(gray_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, grayscale_3channel_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # edge detection of images
        edges = cv2.Canny(img_gray, 50, 150)
        dilated_edges = cv2.dilate(edges, None, iterations=2)
        mask = cv2.threshold(dilated_edges, 127, 255, cv2.THRESH_BINARY_INV)[1]

        # grayscale denoising of images
        gray_denoised_img = cv2.bilateralFilter(img_gray, d=15, sigmaColor=30, sigmaSpace=75)
        gray_denoised_img = cv2.bitwise_and(gray_denoised_img, mask) + cv2.bitwise_and(img_gray, cv2.bitwise_not(mask))
        gray_3channel_denoised_img = cv2.cvtColor(gray_denoised_img, cv2.COLOR_GRAY2BGR)

        gray_denoising = os.path.join(output_folder, "gray_denoising")
        if not os.path.exists(gray_denoising):
            os.makedirs(gray_denoising)
        output_path = os.path.join(gray_denoising, os.path.basename(img_path))
        cv2.imwrite(output_path, gray_3channel_denoised_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # color denoising of images
        color_denoised_img = cv2.bilateralFilter(img_color, d=15, sigmaColor=30, sigmaSpace=75)
        color_denoised_img = cv2.bitwise_and(color_denoised_img, color_denoised_img,mask=mask) + cv2.bitwise_and(img_color, img_color,mask=cv2.bitwise_not(mask))

        color_denoising = os.path.join(output_folder, "color_denoising")
        if not os.path.exists(color_denoising):
            os.makedirs(color_denoising)
        output_path = os.path.join(color_denoising, os.path.basename(img_path))
        cv2.imwrite(output_path, color_denoised_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # scaling the image to scale FACTOR 0.9
        scaled_img = cv2.resize(img_color, (int(h*0.9), int(w*0.9)))

        scale_folder = os.path.join(output_folder, "scaled")
        if not os.path.exists(scale_folder):
            os.makedirs(scale_folder)
        output_path = os.path.join(scale_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, scaled_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # cropping the image with crop size (10,10)
        cropped_img = img_color[10:(h-10), 10:(w-10)]

        crop_folder = os.path.join(output_folder, "cropped")
        if not os.path.exists(crop_folder):
            os.makedirs(crop_folder)
        output_path = os.path.join(crop_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, cropped_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # adding noise to image at noice level 10
        noise = np.random.randint(0,10,img_color.shape, dtype="uint8")
        noisy_img =cv2.add(img_color, noise)

        noise_folder = os.path.join(output_folder, "noisy")
        if not os.path.exists(noise_folder):
            os.makedirs(noise_folder)
        output_path = os.path.join(noise_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, noisy_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

        # perspective transformation of images
        src_points = np.float32([[0, 0], [w-1, 0], [0, h-1], [w-1, h-1]])
        dst_points = src_points + np.random.normal(0, 5, src_points.shape).astype(np.float32)
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        trans_img = cv2.warpPerspective(img_color, M, (w, h))

        perspective_folder = os.path.join(output_folder, "perspective_transformed")
        if not os.path.exists(perspective_folder):
            os.makedirs(perspective_folder)
        output_path = os.path.join(perspective_folder, os.path.basename(img_path))
        cv2.imwrite(output_path, trans_img)
        train_image_paths.append(output_path)
        train_labels.append(label)

    return train_image_paths, train_labels
