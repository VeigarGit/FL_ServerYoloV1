import os
import shutil
import random

def split_dataset(dataset_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    random.seed(seed)

    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")

    # Get all image files
    images = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort()

    # Shuffle
    random.shuffle(images)

    n_total = len(images)
    n_train = int(train_ratio * n_total)
    n_val = int(val_ratio * n_total)
    # Remaining goes to test
    n_test = n_total - n_train - n_val

    splits = {
        "train": images[:n_train],
        "val": images[n_train:n_train+n_val],
        "test": images[n_train+n_val:]
    }

    for split, split_images in splits.items():
        split_img_dir = os.path.join(output_dir, split, "images")
        split_lbl_dir = os.path.join(output_dir, split, "labels")
        os.makedirs(split_img_dir, exist_ok=True)
        os.makedirs(split_lbl_dir, exist_ok=True)

        for img_file in split_images:
            # Copy image
            shutil.copy(os.path.join(images_dir, img_file), split_img_dir)

            # Copy corresponding label if exists
            label_file = os.path.splitext(img_file)[0] + ".txt"
            src_label = os.path.join(labels_dir, label_file)
            if os.path.exists(src_label):
                shutil.copy(src_label, split_lbl_dir)

    print(f"Dataset split complete: {n_train} train, {n_val} val, {n_test} test")

# Example usage
split_dataset("datasets/P33-1/train", "datasets/P-All")