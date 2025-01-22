from PIL import Image
import numpy as np
import os
from sklearn.utils import shuffle


class ImageLoader:
    def __init__(self, split_size=1):
        self.split_size = split_size

    def load(self, folder_path):
        images = []
        labels = []

        for label in range(10):
            label_folder = os.path.join(folder_path, str(label))
            print(f"Loading images from folder {label}...")

            if not os.path.exists(label_folder):
                print(f"Folder {label} does not exist.")
                continue

            image_count = 0
            for img_file in os.listdir(label_folder):
                img_path = os.path.join(label_folder, img_file)

                try:
                    img = Image.open(img_path).convert('L')
                    img = img.resize((28, 28))
                    img = np.array(img)

                    images.append(img)
                    labels.append(label)
                    image_count += 1
                    print(f"Loaded image: {img_file} from folder {label}")

                except Exception as e:
                    print(f"Error loading image {img_file}: {e}")

            print(f"Number of images loaded from folder {label}: {image_count}")

        print(f"Total images loaded: {len(images)}")

        unique, counts = np.unique(labels, return_counts=True)
        print("Loaded data distribution:", dict(zip(unique, counts)))

        images, labels = shuffle(images, labels, random_state=42)

        train_size = int(len(images) * self.split_size)
        train_images = images[:train_size]
        train_labels = labels[:train_size]
        val_images = images[train_size:]
        val_labels = labels[train_size:]

        print(f"Training images: {len(train_images)}, Validation images: {len(val_images)}")

        unique_train, counts_train = np.unique(train_labels, return_counts=True)
        print(f"Training set labels distribution: {dict(zip(unique_train, counts_train))}")

        unique_val, counts_val = np.unique(val_labels, return_counts=True)
        print(f"Validation set labels distribution: {dict(zip(unique_val, counts_val))}")

        train_images = np.array(train_images)
        train_labels = np.array(train_labels)
        val_images = np.array(val_images)
        val_labels = np.array(val_labels)

        return train_images, train_labels, val_images, val_labels
