# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
from PIL import Image, ImageDraw
import json
import shutil
from pathlib import Path
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 3
dir_path = "COCO128/"


class SyntheticCOCODataset:
    def __init__(self, num_images=128, img_size=640):
        self.num_images = num_images
        self.img_size = img_size
        self.num_classes = 80  # COCO has 80 classes
        
    def generate_image(self, image_id):
        """Generate a synthetic image with random objects"""
        # Create random background
        img = Image.new('RGB', (self.img_size, self.img_size), 
                       color=tuple(np.random.randint(0, 255, 3)))
        draw = ImageDraw.Draw(img)
        
        # Generate random number of objects (1-5 per image)
        num_objects = np.random.randint(1, 6)
        annotations = []
        
        for _ in range(num_objects):
            # Random class
            class_id = np.random.randint(0, self.num_classes)
            
            # Random bounding box (normalized coordinates for YOLO)
            w = np.random.uniform(0.1, 0.4)
            h = np.random.uniform(0.1, 0.4)
            x_center = np.random.uniform(w/2, 1 - w/2)
            y_center = np.random.uniform(h/2, 1 - h/2)
            
            # Convert to absolute coordinates for drawing
            x1 = int((x_center - w/2) * self.img_size)
            y1 = int((y_center - h/2) * self.img_size)
            x2 = int((x_center + w/2) * self.img_size)
            y2 = int((y_center + h/2) * self.img_size)
            
            # Draw rectangle
            color = tuple(np.random.randint(0, 255, 3))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            # YOLO format: class_id x_center y_center width height
            annotations.append([class_id, x_center, y_center, w, h])
        
        return img, annotations
    
    def generate_dataset(self):
        """Generate the complete synthetic COCO128 dataset"""
        images = []
        labels = []
        image_info = []
        
        for i in range(self.num_images):
            img, annotations = self.generate_image(i)
            images.append(np.array(img))
            labels.append(annotations)
            image_info.append({
                'id': i,
                'file_name': f'image_{i:06d}.jpg',
                'height': self.img_size,
                'width': self.img_size
            })
        
        return np.array(images), labels, image_info


def coco_to_federated_format(images, labels):
    """Convert COCO format to federated learning compatible format"""
    dataset_image = []
    dataset_label = []
    
    for img, img_labels in zip(images, labels):
        dataset_image.append(img)
        # For object detection, we store both image and its annotations
        dataset_label.append({
            'image': img,
            'annotations': img_labels,
            'num_objects': len(img_labels)
        })
    
    return np.array(dataset_image), dataset_label


def save_yolo_format(data, save_dir, client_id=None):
    """Save data in YOLO format"""
    if client_id is not None:
        images_dir = Path(save_dir) / 'images' / f'{client_id}'
        labels_dir = Path(save_dir) / 'labels' / f'{client_id}'
    else:
        images_dir = Path(save_dir) / 'images'
        labels_dir = Path(save_dir) / 'labels'
    
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    for i, (img_data, label_data) in enumerate(data):
        # Save image
        img = Image.fromarray(img_data)
        img.save(images_dir / f'image_{i:06d}.jpg')
        
        # Save label file (YOLO format)
        label_file = labels_dir / f'image_{i:06d}.txt'
        with open(label_file, 'w') as f:
            for annotation in label_data['annotations']:
                line = ' '.join(map(str, annotation)) + '\n'
                f.write(line)


def generate_coco_dataset(dir_path, num_clients, niid, balance, partition):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        print("Dataset already exists. Skipping generation.")
        return

    print("Generating synthetic COCO128 dataset...")
    
    # Generate synthetic COCO dataset
    coco_generator = SyntheticCOCODataset(num_images=128)
    images, labels, image_info = coco_generator.generate_dataset()
    
    # Convert to federated learning format
    dataset_image, dataset_label = coco_to_federated_format(images, labels)
    
    print(f"Generated {len(dataset_image)} images with {coco_generator.num_classes} classes")
    
    # For object detection, we need to adapt the separation logic
    # We'll separate based on the dominant class in each image
    def get_dominant_class(annotations):
        if not annotations:
            return 0
        class_counts = {}
        for ann in annotations:
            class_id = ann[0]
            class_counts[class_id] = class_counts.get(class_id, 0) + 1
        return max(class_counts.items(), key=lambda x: x[1])[0]
    
    # Extract dominant classes for separation
    dominant_classes = [get_dominant_class(label_data['annotations']) for label_data in dataset_label]
    
    # Convert to format compatible with separate_data function
    y_dominant = np.array(dominant_classes)
    
    num_classes = coco_generator.num_classes
    print(f'Number of classes: {num_classes}')

    # Separate data for federated learning
    X, y, statistic = separate_data((dataset_image, y_dominant), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=16)  # More classes per client for detection
    
    # Convert back to our object detection format
    def reconstruct_detection_data(X_separated, y_separated, original_labels):
        reconstructed_data = []
        for client_data, client_dominant in zip(X_separated, y_separated):
            client_detection_data = []
            for img, dom_class in zip(client_data, client_dominant):
                # Find the corresponding original label
                for orig_img, orig_label in zip(dataset_image, dataset_label):
                    if np.array_equal(img, orig_img):
                        client_detection_data.append((img, orig_label))
                        break
            reconstructed_data.append(client_detection_data)
        return reconstructed_data
    
    # Split data (for object detection, we need custom splitting)
    train_ratio = 0.8
    train_data = []
    test_data = []
    
    for client_idx in range(len(X)):
        client_images = X[client_idx]
        client_labels = [dataset_label[i] for i in range(len(dataset_label)) 
                        if any(np.array_equal(client_images[j], dataset_image[i]) 
                              for j in range(len(client_images)))]
        
        # Split client data
        split_idx = int(len(client_images) * train_ratio)
        client_train_data = list(zip(client_images[:split_idx], client_labels[:split_idx]))
        client_test_data = list(zip(client_images[split_idx:], client_labels[split_idx:]))
        
        train_data.append(client_train_data)
        test_data.append(client_test_data)

    # Save in YOLO format
    print("Saving data in YOLO format...")
    for client_id in range(num_clients):
        save_yolo_format(train_data[client_id], train_path, client_id)
        save_yolo_format(test_data[client_id], test_path, client_id)
    
    # Also save original full dataset
    save_yolo_format(list(zip(dataset_image, dataset_label)), dir_path + "full_dataset")
    
    # Save dataset configuration
    dataset_config = {
        'num_clients': num_clients,
        'niid': niid,
        'balance': balance,
        'partition': partition,
        'num_classes': num_classes,
        'img_size': 640,
        'total_images': len(dataset_image),
        'yolo_format': True
    }
    
    with open(config_path, 'w') as f:
        json.dump(dataset_config, f, indent=4)
    
    print("COCO128-like dataset generation completed!")
    print(f"Train data saved to: {train_path}")
    print(f"Test data saved to: {test_path}")


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    generate_coco_dataset(dir_path, num_clients, niid, balance, partition)