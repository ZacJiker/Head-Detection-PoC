import os
import torch
from torch.utils.data import Dataset
import cv2
import yaml

class YoloDataset(Dataset):
    def __init__(self, data_yaml, img_size=640, mode="train"):
        """
        YOLO-formatted Dataset loader.

        Args:
            data_yaml (str): Path to data.yaml file.
            img_size (int): Target image size for resizing.
            mode (str): Mode can be 'train', 'val', or 'test'.
        """
        self.img_size = img_size
        self.mode = mode

        # Load paths from data.yaml
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        
        if mode == "train":
            self.img_dir = data_config['train']
        elif mode == "val":
            self.img_dir = data_config['val']
        elif mode == "test":
            self.img_dir = data_config['test']
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")
        
        # Infer label directory from image directory
        self.label_dir = self.img_dir.replace("images", "labels")
        
        # List all images and labels
        self.images = sorted([os.path.join(self.img_dir, img) for img in os.listdir(self.img_dir) if img.endswith(('.jpg', '.png'))])
        self.labels = sorted([os.path.join(self.label_dir, lbl) for lbl in os.listdir(self.label_dir) if lbl.endswith('.txt')])

        assert len(self.images) == len(self.labels), "Mismatch between images and labels count"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize image
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0  # Normalize to [0, 1]
        img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)  # HWC -> CHW

        # Load labels
        label_path = self.labels[idx]
        bboxes = []
        with open(label_path, 'r') as f:
            for line in f.readlines():
                class_id, x, y, w, h = map(float, line.strip().split())
                # Convert from normalized YOLO format to pixel coordinates
                x_min = (x - w / 2) * self.img_size
                y_min = (y - h / 2) * self.img_size
                x_max = (x + w / 2) * self.img_size
                y_max = (y + h / 2) * self.img_size
                bboxes.append([x_min, y_min, x_max, y_max, class_id])

        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        # Create mask (for valid bboxes)
        mask = torch.ones(len(bboxes), dtype=torch.float32)

        return img, bboxes, mask
