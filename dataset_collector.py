import os
import cv2
import json
import numpy as np
from datetime import datetime
from card_constants import get_valid_label


class DatasetCollector:
    def __init__(self, base_path="dataset"):
        self.base_path = base_path
        self.images_path = os.path.join(base_path, "images")
        self.labels_path = os.path.join(base_path, "labels")
        self.metadata_file = os.path.join(base_path, "metadata.json")
        self.setup_directories()
        self.load_metadata()

    def setup_directories(self):
        for path in [self.images_path, self.labels_path]:
            if not os.path.exists(path):
                os.makedirs(path)

    def load_metadata(self):
        if os.path.exists(self.metadata_file):
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {"total_images": 0, "labels": {}}

    def save_metadata(self):
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=4)

    def apply_random_variation(self, image, binary):
        """
        Menerapkan variasi acak pada gambar untuk augmentasi data
        """
        # Random brightness variation
        brightness = np.random.uniform(0.8, 1.2)
        varied_image = cv2.convertScaleAbs(image, alpha=brightness, beta=0)

        # Random rotation (slight)
        angle = np.random.uniform(-5, 5)
        height, width = image.shape[:2]
        center = (width/2, height/2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        varied_image = cv2.warpAffine(
            varied_image, rotation_matrix, (width, height))
        varied_binary = cv2.warpAffine(
            binary, rotation_matrix, (width, height))

        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 5, varied_image.shape).astype(np.uint8)
            varied_image = cv2.add(varied_image, noise)

        return varied_image, varied_binary

    def save_card_copies(self, original, binary, label, num_copies=50):
        """
        Menyimpan multiple copy dari kartu dengan variasi
        """
        # Validasi label
        valid_label = get_valid_label(label)
        if not valid_label:
            print(f"Invalid card label: {label}")
            return None

        # Generate base filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{valid_label}_{timestamp}"

        saved_files = []

        # Simpan multiple copy dengan variasi
        for i in range(num_copies):
            # Apply random variations
            varied_original, varied_binary = self.apply_random_variation(
                original, binary)

            # Generate filename untuk copy ini
            img_name = f"{base_name}_{i+1:03d}"

            # Simpan gambar
            cv2.imwrite(os.path.join(self.images_path, f"{
                        img_name}_original.jpg"), varied_original)
            cv2.imwrite(os.path.join(self.images_path, f"{
                        img_name}_binary.jpg"), varied_binary)

            # Simpan label
            label_data = {
                "image_name": img_name,
                "label": valid_label,
                "timestamp": timestamp,
                "copy_number": i+1
            }

            with open(os.path.join(self.labels_path, f"{img_name}.json"), 'w') as f:
                json.dump(label_data, f, indent=4)

            saved_files.append(img_name)

        # Update metadata
        self.metadata["total_images"] += num_copies
        if valid_label not in self.metadata["labels"]:
            self.metadata["labels"][valid_label] = 0
        self.metadata["labels"][valid_label] += num_copies

        self.save_metadata()
        return saved_files
