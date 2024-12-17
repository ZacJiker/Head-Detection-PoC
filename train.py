import torch
import torch.nn as nn
import torch.optim as optim
import logging

from torch.utils.data import DataLoader
from tqdm import tqdm
from bbox_dataset import BBoxDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

if torch.cuda.is_available():
    torch.cuda.empty_cache()

class BBoxDetectionModel(nn.Module):
    def __init__(self, num_bboxes=100):
        super(BBoxDetectionModel, self).__init__()
        self.num_bboxes = num_bboxes
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.bbox_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * (640 // 4) * (640 // 4), 1024),
            nn.ReLU(),
            nn.Linear(1024, num_bboxes * 4),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bbox_head(x)
        return x.view(x.size(0), self.num_bboxes, 4)


def bbox_loss(predicted, target, mask):
    """
    Calculate bounding box loss.
    
    Args:
        predicted (torch.Tensor): Predicted bounding boxes.
        target (torch.Tensor): Ground truth bounding boxes.
        mask (torch.Tensor): Mask indicating valid bounding boxes.
    
    Returns:
        torch.Tensor: Loss.
    """
    l1_loss = nn.functional.smooth_l1_loss(predicted * mask.unsqueeze(-1), target * mask.unsqueeze(-1))
    return l1_loss


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    logging.info("Starting training...")
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for images, bboxes, mask in progress_bar:
            images, bboxes, mask = images.to(device), bboxes.to(device), mask.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, bboxes, mask)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        logging.info(f"Epoch {epoch+1} completed. Average Loss: {running_loss / len(train_loader):.4f}")
    logging.info("Training completed!")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    dataset = BBoxDataset(
        annotations_file="/Users/baptiste/Documents/Developpement/Aix Ynov Campus Sas/Head-Dectection-PoC/datas/head-detection-dataset-v1/train/_annotations.coco.json",
        images_dir="/Users/baptiste/Documents/Developpement/Aix Ynov Campus Sas/Head-Dectection-PoC/datas/head-detection-dataset-v1/train",
        num_bboxes=100
    )
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    model = BBoxDetectionModel(num_bboxes=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)
    criterion = bbox_loss

    train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)
    torch.save(model.state_dict(), "bbox_detection_model.pth")
    logging.info("Model saved as bbox_detection_model.pth")
