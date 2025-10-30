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
import json
import shutil
from pathlib import Path
from PIL import Image, ImageDraw
import yaml
import zipfile
import urllib.request
from utils.yolo_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)
num_clients = 5
dir_path = "COCO128/"


class COCO128Loader:
    def __init__(self, dataset_path, img_size=640):
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / 'images' / 'train2017'
        self.labels_path = self.dataset_path / 'labels' / 'train2017'
        self.img_size = img_size
        
    def load_dataset(self):
        """Load COCO128 dataset - return lists instead of numpy arrays for variable size images"""
        images = []
        labels = []
        image_info = []
        
        # Find all image files
        image_files = list(self.images_path.glob('*.jpg'))
        
        print(f"Found {len(image_files)} images in dataset")
        
        for img_path in image_files:
            try:
                # Load image
                img = Image.open(img_path)
                original_size = img.size  # (width, height)
                
                # Resize image to consistent size for processing
                img_resized = img.resize((self.img_size, self.img_size))
                img_array = np.array(img_resized)
                
                # Load corresponding label file
                label_path = self.labels_path / f'{img_path.stem}.txt'
                annotations = []
                
                if label_path.exists():
                    with open(label_path, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) == 5:
                                class_id = int(parts[0])
                                x_center = float(parts[1])
                                y_center = float(parts[2])
                                width = float(parts[3])
                                height = float(parts[4])
                                annotations.append([class_id, x_center, y_center, width, height])
                
                images.append(img_array)
                labels.append({
                    'image_path': str(img_path),
                    'annotations': annotations,
                    'num_objects': len(annotations),
                    'original_size': original_size,
                    'resized_size': (self.img_size, self.img_size)
                })
                image_info.append({
                    'file_name': img_path.name,
                    'height': img.height,
                    'width': img.width
                })
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        return images, labels, image_info  # Return as lists, not numpy arrays
    
    def get_class_distribution(self, labels):
        """Get distribution of classes in the dataset"""
        class_counts = {}
        for label_data in labels:
            for annotation in label_data['annotations']:
                class_id = annotation[0]
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
        return class_counts


def coco128_to_federated_format(images, labels):
    """Convert COCO128 format to federated learning compatible format"""
    # Return as lists since images have different sizes
    dataset_image = images  # Already a list of numpy arrays
    dataset_label = []
    
    for i, img_labels in enumerate(labels):
        dataset_label.append({
            'image_index': i,
            'annotations': img_labels['annotations'],
            'num_objects': img_labels['num_objects'],
            'image_path': img_labels['image_path'],
            'original_size': img_labels['original_size'],
            'resized_size': img_labels['resized_size']
        })
    
    return dataset_image, dataset_label


def save_yolo_format(data, save_dir, client_id=None, copy_original=True):
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
        # Check if label_data is a dictionary (with image_path) or just annotations
        if isinstance(label_data, dict) and 'image_path' in label_data and copy_original:
            # Copy original image file
            original_path = Path(label_data['image_path'])
            new_path = images_dir / original_path.name
            shutil.copy2(original_path, new_path)
            
            # Copy original label file
            label_original_path = Path(label_data['image_path'].replace('images', 'labels')).with_suffix('.txt')
            label_new_path = labels_dir / original_path.with_suffix('.txt').name
            if label_original_path.exists():
                shutil.copy2(label_original_path, label_new_path)
        else:
            # Save resized image
            img = Image.fromarray(img_data)
            img_filename = f'image_{i:06d}.jpg'
            img.save(images_dir / img_filename)
            
            # Save label file (YOLO format)
            label_filename = f'image_{i:06d}.txt'
            label_file = labels_dir / label_filename
            
            # Handle both dict and list formats for annotations
            if isinstance(label_data, dict):
                annotations = label_data['annotations']
            else:
                # Assume label_data is the dominant class or just annotations
                annotations = label_data if isinstance(label_data, list) else []
            
            with open(label_file, 'w') as f:
                for annotation in annotations:
                    line = ' '.join(map(str, annotation)) + '\n'
                    f.write(line)


def create_yolo_dataset_yaml(save_path, num_classes):
    """Create dataset.yaml file for YOLO training"""
    dataset_config = {
        'path': str(Path(save_path).absolute()),
        'train': 'train/images',
        'val': 'test/images',
        'nc': num_classes,
        'names': [f'class_{i}' for i in range(num_classes)]
    }
    
    yaml_path = Path(save_path) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False, sort_keys=False)
    
    return yaml_path


def get_dominant_class(annotations):
    """Get the dominant class in annotations"""
    if not annotations:
        return 0
    class_counts = {}
    for ann in annotations:
        class_id = ann[0]
        class_counts[class_id] = class_counts.get(class_id, 0) + 1
    if not class_counts:
        return 0
    return max(class_counts.items(), key=lambda x: x[1])[0]


def download_coco128_simple(download_path='coco128'):
    """Download COCO128 dataset directly"""
    dataset_path = Path(download_path)
    
    if dataset_path.exists():
        print(f"COCO128 already exists at {dataset_path}")
        return True
    
    print("Downloading COCO128 dataset...")
    
    # URL do COCO128 do Ultralytics
    coco128_url = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128.zip"
    
    try:
        # Download the file
        print(f"Downloading from {coco128_url}...")
        zip_path = 'coco128.zip'
        urllib.request.urlretrieve(coco128_url, zip_path)
        
        # Extract
        print("Extracting dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(download_path)
        
        # Move contents if needed
        extracted_path = Path(download_path) / 'coco128'
        if extracted_path.exists():
            # Move all contents to main download_path
            for item in extracted_path.iterdir():
                shutil.move(str(item), str(download_path))
            extracted_path.rmdir()
        
        # Cleanup
        os.remove(zip_path)
        
        print("COCO128 downloaded successfully!")
        return True
        
    except Exception as e:
        print(f"Error downloading COCO128: {e}")
        print("\nAlternative download methods:")
        print("1. Download manually from: https://ultralytics.com/assets/coco128.zip")
        print("2. Or run: wget https://ultralytics.com/assets/coco128.zip && unzip coco128.zip")
        return False


def create_mini_coco128_dataset(output_path='coco128', num_images=128):
    """Create a mini COCO128-like dataset if download fails"""
    print(f"Creating mini COCO128-like dataset with {num_images} images...")
    
    dataset_path = Path(output_path)
    images_path = dataset_path / 'images' / 'train2017'
    labels_path = dataset_path / 'labels' / 'train2017'
    
    images_path.mkdir(parents=True, exist_ok=True)
    labels_path.mkdir(parents=True, exist_ok=True)
    
    num_classes = 80  # COCO has 80 classes
    
    for i in range(num_images):
        # Create synthetic image
        img = Image.new('RGB', (640, 480), color=tuple(np.random.randint(0, 255, 3)))
        draw = ImageDraw.Draw(img)
        
        # Generate random objects
        num_objects = np.random.randint(1, 4)
        annotations = []
        
        for _ in range(num_objects):
            class_id = np.random.randint(0, num_classes)
            w = np.random.uniform(0.1, 0.3)
            h = np.random.uniform(0.1, 0.3)
            x_center = np.random.uniform(w/2, 1 - w/2)
            y_center = np.random.uniform(h/2, 1 - h/2)
            
            # Draw rectangle
            x1 = int((x_center - w/2) * 640)
            y1 = int((y_center - h/2) * 480)
            x2 = int((x_center + w/2) * 640)
            y2 = int((y_center + h/2) * 480)
            
            color = tuple(np.random.randint(0, 255, 3))
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
            
            annotations.append([class_id, x_center, y_center, w, h])
        
        # Save image
        img.save(images_path / f'image_{i:06d}.jpg')
        
        # Save labels
        with open(labels_path / f'image_{i:06d}.txt', 'w') as f:
            for ann in annotations:
                f.write(' '.join(map(str, ann)) + '\n')
    
    print(f"Mini dataset created at {output_path}")
    return True


def separate_data_detection(dataset, num_clients, num_classes, niid=False, balance=False, partition=None, class_per_client=2):
    """
    Separate object detection data for federated learning - working with lists
    """
    X, y_dominant = dataset
    
    if not niid:
        # IID case - simple random split
        idxs = list(range(len(X)))
        random.shuffle(idxs)
        batch_idxs = np.array_split(idxs, num_clients)
        
        client_data = [[X[i] for i in idxs] for idxs in batch_idxs]
        client_labels = [[y_dominant[i] for i in idxs] for idxs in batch_idxs]
        
    else:
        # Non-IID case - separate by dominant class
        from collections import defaultdict
        class_indices = defaultdict(list)
        
        for idx, dominant_class in enumerate(y_dominant):
            class_indices[dominant_class].append(idx)
        
        client_data = [[] for _ in range(num_clients)]
        client_labels = [[] for _ in range(num_clients)]
        
        # Assign classes to clients
        classes_per_client = max(1, class_per_client)
        all_classes = list(range(num_classes))
        random.shuffle(all_classes)
        
        for client_id in range(num_clients):
            start_idx = (client_id * classes_per_client) % num_classes
            client_classes = all_classes[start_idx:start_idx + classes_per_client]
            
            for class_id in client_classes:
                if class_id in class_indices:
                    indices = class_indices[class_id]
                    # Take a subset of images for this class
                    n_samples = min(len(indices), max(1, len(indices) // num_clients))
                    selected_indices = random.sample(indices, n_samples)
                    
                    for idx in selected_indices:
                        client_data[client_id].append(X[idx])
                        client_labels[client_id].append(y_dominant[idx])
    
    # Calculate statistics
    statistic = [[] for _ in range(num_clients)]
    for client_id in range(num_clients):
        client_y = client_labels[client_id]
        for class_id in range(num_classes):
            count = sum(1 for dominant_class in client_y if dominant_class == class_id)
            if count > 0:
                statistic[client_id].append((class_id, count))
    
    return client_data, client_labels, statistic


def generate_coco128_federated(dir_path, num_clients, niid, balance, partition, coco128_path='coco128'):
    """
    Generate federated COCO128 dataset
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "config.json"
    train_path = dir_path + "train/"
    test_path = dir_path + "test/"

    if check(config_path, train_path, test_path, num_clients, niid, balance, partition):
        print("Dataset already exists. Skipping generation.")
        return

    print("Loading COCO128 dataset...")
    
    # Check if COCO128 exists, if not create a mini version
    if not os.path.exists(coco128_path):
        print("COCO128 not found. Creating mini dataset...")
        create_mini_coco128_dataset(coco128_path)
    
    # Load COCO128 dataset
    coco_loader = COCO128Loader(coco128_path, img_size=640)
    images, labels, image_info = coco_loader.load_dataset()
    
    if len(images) == 0:
        print("No images found in dataset. Creating mini dataset...")
        create_mini_coco128_dataset(coco128_path)
        coco_loader = COCO128Loader(coco128_path, img_size=640)
        images, labels, image_info = coco_loader.load_dataset()
    
    # Get class distribution
    class_distribution = coco_loader.get_class_distribution(labels)
    num_classes = len(class_distribution) if class_distribution else 80
    print(f"Loaded {len(images)} images with {num_classes} classes")
    print(f"Class distribution: {class_distribution}")
    
    # Convert to federated learning format
    dataset_image, dataset_label = coco128_to_federated_format(images, labels)
    
    # Extract dominant classes for separation
    dominant_classes = [get_dominant_class(label_data['annotations']) for label_data in dataset_label]
    
    print(f'Number of classes: {num_classes}')
    print(f'Total images: {len(dataset_image)}')

    # Separate data for federated learning - using our custom function
    X, y_dominant, statistic = separate_data_detection((dataset_image, dominant_classes), num_clients, num_classes, 
                                    niid, balance, partition, class_per_client=max(1, num_classes//num_clients))
    
    # Split data into train and test
    train_ratio = 0.8
    train_data = []
    test_data = []
    
    # Create a mapping from image to its full label data
    image_to_label = {}
    for i, (img, label_dict) in enumerate(zip(dataset_image, dataset_label)):
        image_to_label[i] = label_dict
    
    for client_idx in range(len(X)):
        client_images = X[client_idx]
        client_dominant_classes = y_dominant[client_idx]
        
        # Create paired data with full label information
        client_paired_data = []
        for i, (img, dom_class) in enumerate(zip(client_images, client_dominant_classes)):
            # Find the corresponding full label data
            for idx, original_img in enumerate(dataset_image):
                if np.array_equal(img, original_img):
                    client_paired_data.append((img, dataset_label[idx]))
                    break
            else:
                # If not found, create a minimal label dict
                client_paired_data.append((img, {
                    'annotations': [],
                    'image_path': f'client_{client_idx}_image_{i}.jpg'
                }))
        
        # Split client data
        split_idx = int(len(client_paired_data) * train_ratio)
        random.shuffle(client_paired_data)
        
        client_train_data = client_paired_data[:split_idx]
        client_test_data = client_paired_data[split_idx:]
        
        train_data.append(client_train_data)
        test_data.append(client_test_data)
        
        print(f"Client {client_idx}: {len(client_train_data)} train, {len(client_test_data)} test images")

    # Save in YOLO format
    print("Saving data in YOLO format...")
    
    # Save client data
    for client_id in range(num_clients):
        save_yolo_format(train_data[client_id], train_path, client_id, copy_original=True)
        save_yolo_format(test_data[client_id], test_path, client_id, copy_original=True)
    
    # Create YOLO dataset configuration
    create_yolo_dataset_yaml(dir_path, num_classes)
    
    # Save dataset configuration
    dataset_config = {
        'num_clients': num_clients,
        'niid': niid,
        'balance': balance,
        'partition': partition,
        'num_classes': num_classes,
        'total_images': len(dataset_image),
        'class_distribution': class_distribution,
        'yolo_format': True,
        'original_dataset': 'COCO128'
    }
    
    with open(config_path, 'w') as f:
        json.dump(dataset_config, f, indent=4)
    
    # Print statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total clients: {num_clients}")
    print(f"NIID: {niid}")
    print(f"Balance: {balance}")
    print(f"Partition: {partition}")
    print(f"Total images: {len(dataset_image)}")
    print(f"Number of classes: {num_classes}")
    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")
    print("COCO128 federated dataset generation completed!")

# Adicione esta função ao seu generate_coco128.py e chame-a no final

def generate_client_yamls_after_creation(dataset_path, num_clients, num_classes):
    """
    Generate YAML files after dataset creation - to be called at the end of generate_coco128_federated
    """
    from pathlib import Path
    import yaml
    
    dataset_path = Path(dataset_path)
    
    # COCO class names
    coco_names = [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    
    # Adjust names to match num_classes
    if num_classes <= 80:
        names = coco_names[:num_classes]
    else:
        names = coco_names + [f'class_{i}' for i in range(80, num_classes)]
    
    names_dict = {i: name for i, name in enumerate(names)}
    
    # Create global YAML
    global_config = {
        'path': str(dataset_path.absolute()),
        'train': 'train/images',
        'val': 'test/images',
        'nc': num_classes,
        'names': names_dict
    }
    
    with open(dataset_path / 'dataset.yaml', 'w') as f:
        yaml.dump(global_config, f, default_flow_style=False, sort_keys=False)
    
    # Create client YAMLs
    for client_id in range(num_clients):
        client_config = {
            'path': str(dataset_path.absolute()),
            'train': f'train/images/{client_id}',
            'val': f'test/images/{client_id}',
            'nc': num_classes,
            'names': names_dict
        }
        
        with open(dataset_path / f'{client_id}.yaml', 'w') as f:
            yaml.dump(client_config, f, default_flow_style=False, sort_keys=False)
    
    print(f"Generated YAML files for {num_clients} clients")

# No final da função generate_coco128_federated, adicione:
# generate_client_yamls_after_creation(dir_path, num_clients, num_classes)
if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) > 1:
        niid = True if sys.argv[1] == "noniid" else False
    else:
        niid = False
        
    if len(sys.argv) > 2:
        balance = True if sys.argv[2] == "balance" else False
    else:
        balance = False
        
    if len(sys.argv) > 3:
        partition = sys.argv[3] if sys.argv[3] != "-" else None
    else:
        partition = None
    
    # Try to download COCO128, if fails create mini dataset
    coco128_path = "coco128"
    if not download_coco128_simple(coco128_path):
        print("Download failed. Creating mini COCO128 dataset...")
        create_mini_coco128_dataset(coco128_path)
    
    generate_coco128_federated(dir_path, num_clients, niid, balance, partition, coco128_path)
    generate_client_yamls_after_creation(dir_path, num_clients, 71)