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


def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluate the model on the test set.
    """
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for images, bboxes, mask in test_loader:
            images, bboxes, mask = images.to(device), bboxes.to(device), mask.to(device)
            outputs = model(images)
            loss = criterion(outputs, bboxes, mask)
            test_loss += loss.item()
    return test_loss / len(test_loader)


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    logging.info("Starting training...")
    prev_test_loss = float('inf')
    overfit_counter = 0

    for epoch in range(num_epochs):
        model.train()
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

        avg_train_loss = running_loss / len(train_loader)
        avg_test_loss = evaluate_model(model, test_loader, criterion, device)

        logging.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}")

        # Overfitting detection
        if avg_test_loss > prev_test_loss:
            overfit_counter += 1
            logging.warning(f"Overfitting alert: Test loss increased! (Count: {overfit_counter})")
        else:
            overfit_counter = 0
        
        prev_test_loss = avg_test_loss
        
        if overfit_counter >= 3:
            logging.error("Training stopped due to overfitting detection!")
            break

    logging.info("Training completed!")


if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # Train dataset
    train_dataset = BBoxDataset(
        annotations_file="/path/to/train/_annotations.coco.json",
        images_dir="/path/to/train",
        num_bboxes=100
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)

    # Test dataset
    test_dataset = BBoxDataset(
        annotations_file="/path/to/test/_annotations.coco.json",
        images_dir="/path/to/test",
        num_bboxes=100
    )
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)

    model = BBoxDetectionModel(num_bboxes=100).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.)
    criterion = bbox_loss

    train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10)
    torch.save(model.state_dict(), "bbox_detection_model.pth")
    logging.info("Model saved as bbox_detection_model.pth")
