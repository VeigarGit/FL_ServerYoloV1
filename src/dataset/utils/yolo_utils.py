import yaml
from pathlib import Path

def create_yolo_dataset_config(dataset_path, num_classes, class_names=None):
    """Create YOLO dataset configuration file"""
    if class_names is None:
        class_names = [f"class_{i}" for i in range(num_classes)]
    
    config = {
        'path': str(Path(dataset_path).absolute()),
        'train': 'images',
        'val': 'images',  # For simplicity, using same for train/val
        'names': {i: name for i, name in enumerate(class_names)}
    }
    
    # Save dataset.yaml
    with open(Path(dataset_path) / 'dataset.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    return config

def validate_yolo_labels(labels_dir, num_classes):
    """Validate YOLO label files"""
    label_files = list(Path(labels_dir).glob('*.txt'))
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"Invalid line in {label_file}: {line}")
                continue
            
            class_id, x_center, y_center, width, height = map(float, parts)
            
            # Validate ranges
            if not (0 <= class_id < num_classes):
                print(f"Invalid class ID in {label_file}: {class_id}")
            
            for coord in [x_center, y_center, width, height]:
                if not (0 <= coord <= 1):
                    print(f"Invalid coordinate in {label_file}: {coord}")