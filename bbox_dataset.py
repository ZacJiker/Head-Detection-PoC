import json
import torch
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import functional as F

class BBoxDataset(Dataset):
    def __init__(self, annotations_file, images_dir, num_bboxes=100):
        """
        Dataset for bounding box detection.
        
        Args:
            annotations_file (str): Path to the COCO-style JSON file.
            images_dir (str): Path to the directory containing images.
            num_bboxes (int): Maximum number of bounding boxes to include.
        """
        self.num_bboxes = num_bboxes
        self.images_dir = images_dir
        
        # Load annotations from COCO-style JSON
        with open(annotations_file, 'r') as f:
            self.coco_data = json.load(f)

        # Map image IDs to bounding boxes
        self.image_id_to_bboxes = {}
        for annotation in self.coco_data['annotations']:
            image_id = annotation['image_id']
            bbox = annotation['bbox']  # Format: [x_min, y_min, width, height]
            if image_id not in self.image_id_to_bboxes:
                self.image_id_to_bboxes[image_id] = []
            self.image_id_to_bboxes[image_id].append(bbox)

        # Load image metadata
        self.images = self.coco_data['images']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Get image and its bounding boxes.
        
        Args:
            idx (int): Index of the image to retrieve.
        
        Returns:
            torch.Tensor: Image tensor.
            torch.Tensor: Bounding boxes tensor.
            torch.Tensor: Mask tensor indicating valid bounding boxes.
        """
        image_data = self.images[idx]
        image_id = image_data['id']
        file_name = image_data['file_name']

        # Load image
        image_path = f"{self.images_dir}/{file_name}"
        image = Image.open(image_path).convert("RGB")
        image = F.to_tensor(image)  # Convert to tensor

        # Get bounding boxes
        bboxes = self.image_id_to_bboxes.get(image_id, [])
        bboxes, mask = self.prepare_bbox_data(bboxes)

        return image, bboxes, mask

    def prepare_bbox_data(self, bboxes):
        """
        Prepare bounding box data.
        
        Args:
            bboxes (list): List of bounding boxes for an image.
        
        Returns:
            torch.Tensor: Bounding boxes tensor.
            torch.Tensor: Mask tensor.
        """
        bbox_array = torch.zeros((self.num_bboxes, 4))
        mask = torch.zeros(self.num_bboxes)

        for i, bbox in enumerate(bboxes):
            if i >= self.num_bboxes:
                break
            x_min, y_min, width, height = bbox
            bbox_array[i] = torch.tensor([x_min, y_min, x_min + width, y_min + height])
            mask[i] = 1  # Mark as valid

        return bbox_array, mask
