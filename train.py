import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
from yolo_dataset import YoloDataset

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class BBoxDetectionModel(nn.Module):
    def __init__(self, num_bboxes=100):
        super(BBoxDetectionModel, self).__init__()
        self.num_bboxes = num_bboxes

        # Feature extraction layers
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )

        self.bbox_head = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, num_bboxes * 5, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.bbox_head(x)
        batch_size, _, grid_h, grid_w = x.shape
        x = x.view(batch_size, self.num_bboxes, 5, grid_h, grid_w)  # Reshape for bounding boxes
        return x
    

def compute_iou(pred_boxes, target_boxes):
    """
    Compute Intersection over Union (IoU) between predicted and target boxes.

    Args:
        pred_boxes (torch.Tensor): Predicted boxes [num_bboxes, 4].
        target_boxes (torch.Tensor): Target boxes [num_gt_boxes, 4].

    Returns:
        torch.Tensor: IoU matrix [num_bboxes, num_gt_boxes].
    """
    # Expand dimensions for broadcasting
    pred_boxes = pred_boxes.unsqueeze(1)  # Shape: [num_bboxes, 1, 4]
    target_boxes = target_boxes.unsqueeze(0)  # Shape: [1, num_gt_boxes, 4]

    # Compute intersection
    inter_x1 = torch.max(pred_boxes[..., 0], target_boxes[..., 0])
    inter_y1 = torch.max(pred_boxes[..., 1], target_boxes[..., 1])
    inter_x2 = torch.min(pred_boxes[..., 2], target_boxes[..., 2])
    inter_y2 = torch.min(pred_boxes[..., 3], target_boxes[..., 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    # Compute union
    pred_area = (pred_boxes[..., 2] - pred_boxes[..., 0]) * (pred_boxes[..., 3] - pred_boxes[..., 1])
    target_area = (target_boxes[..., 2] - target_boxes[..., 0]) * (target_boxes[..., 3] - target_boxes[..., 1])
    union_area = pred_area + target_area - inter_area

    # Compute IoU
    return inter_area / union_area.clamp(min=1e-6)


def bbox_loss(predicted, targets, masks, iou_threshold=0.5):
    """
    Bounding box loss combining localization and confidence losses.
    Aligns predictions with targets using IoU matching.

    Args:
        predicted (torch.Tensor): Predicted bounding boxes [batch_size, num_bboxes, 5, grid_h, grid_w].
        targets (list of torch.Tensor): List of tensors containing ground truth boxes for each image.
        masks (list of torch.Tensor): List of masks indicating valid bounding boxes.
        iou_threshold (float): IoU threshold for matching predicted boxes to targets.

    Returns:
        torch.Tensor: Average loss for the batch.
    """
    total_loss = 0.0
    batch_size = predicted.shape[0]

    for i in range(batch_size):
        # Predictions for the current image
        pred = predicted[i]  # Shape: [num_bboxes, 5, grid_h, grid_w]
        num_bboxes, _, grid_h, grid_w = pred.shape

        # Flatten the predictions
        pred = pred.permute(0, 2, 3, 1).reshape(-1, 5)  # Shape: [num_bboxes * grid_h * grid_w, 5]
        pred_coords = pred[:, :4]  # Predicted coordinates: x, y, w, h
        pred_conf = pred[:, 4]     # Confidence scores

        # Targets for the current image
        target = targets[i]  # Shape: [num_gt_boxes, 5]
        if target.size(0) == 0:  # Skip if no ground truth boxes
            continue

        target_coords = target[:, :4]  # Target coordinates: x, y, w, h
        target_conf = torch.ones(target.size(0), device=pred.device)  # Ground truth confidence = 1

        # Compute IoU between predictions and targets
        ious = compute_iou(pred_coords, target_coords)

        # Match predictions to targets based on IoU
        best_iou, best_idx = ious.max(dim=1)  # For each prediction, find the best matching target

        # Localization loss: Apply only to predictions with IoU > threshold
        loc_mask = best_iou > iou_threshold

        loc_loss = torch.nn.functional.smooth_l1_loss(
            pred_coords[loc_mask], target_coords[best_idx[loc_mask]], reduction='mean'
        ) if loc_mask.any() else 0.0

        # Confidence loss: Binary Cross-Entropy
        conf_target = (best_iou > iou_threshold).float()  # 1 if matched, 0 otherwise
        conf_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_conf, conf_target, reduction='mean'
        )

        total_loss += loc_loss + conf_loss

    return total_loss / batch_size  # Average loss over the batch


def process_predictions(outputs, confidence_threshold=0.5):
    """
    Process model outputs to extract bounding boxes with confidence scores.

    Args:
        outputs (torch.Tensor): Model predictions [batch_size, num_bboxes, 5, grid_h, grid_w].
        confidence_threshold (float): Minimum confidence score to keep a prediction.

    Returns:
        pred_boxes (list): List of predicted bounding boxes [x_min, y_min, x_max, y_max, confidence].
    """
    pred_boxes = []
    for batch_idx in range(outputs.shape[0]):
        batch_preds = outputs[batch_idx].cpu().numpy()
        for grid in batch_preds.reshape(-1, 5):
            x, y, w, h, confidence = grid
            if confidence > confidence_threshold:
                x_min = x - w / 2
                y_min = y - h / 2
                x_max = x + w / 2
                y_max = y + h / 2
                pred_boxes.append([x_min, y_min, x_max, y_max, confidence])
    return pred_boxes

def process_targets(targets):
    """
    Process target bounding boxes.

    Args:
        targets (list of tensors): List of ground truth bounding boxes per image.

    Returns:
        true_boxes (list): List of ground truth bounding boxes [x_min, y_min, x_max, y_max].
    """
    true_boxes = []
    for target in targets:
        for bbox in target:
            x, y, w, h, _ = bbox.tolist()
            x_min = x - w / 2
            y_min = y - h / 2
            x_max = x + w / 2
            y_max = y + h / 2
            true_boxes.append([x_min, y_min, x_max, y_max])
    return true_boxes

def calculate_map(pred_boxes, true_boxes, iou_threshold=0.5):
    """
    Calculate mean Average Precision (mAP).

    Args:
        pred_boxes (list): List of predicted bounding boxes.
        true_boxes (list): List of ground truth bounding boxes.
        iou_threshold (float): IoU threshold to consider a prediction as positive.

    Returns:
        mAP (float): Mean Average Precision score.
    """
    def iou(box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    aps = []
    for pred, truth in zip(pred_boxes, true_boxes):
        if not truth:
            continue
        tp, fp = 0, 0
        for p in pred:
            if any(iou(p, t) >= iou_threshold for t in truth):
                tp += 1
            else:
                fp += 1
        precision = tp / max(tp + fp, 1)
        recall = tp / len(truth)
        aps.append((precision + recall) / 2)
    return sum(aps) / len(aps) if aps else 0


def evaluate_model(model, test_loader, criterion, device, iou_threshold=0.5):
    """
    Evaluate the model on the test set.

    Args:
        model (torch.nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        criterion (function): Loss function to compute the loss.
        device (torch.device): Device to run the evaluation.
        iou_threshold (float): IoU threshold to determine positive predictions.

    Returns:
        avg_loss (float): Average loss on the test set.
        avg_map (float): Mean Average Precision on the test set.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    all_pred_boxes = []
    all_true_boxes = []

    with torch.no_grad():
        for images, targets, masks in test_loader:
            # Move images and targets to the appropriate device
            images = images.to(device)
            targets = [t.to(device) for t in targets]
            masks = [m.to(device) for m in masks]

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, targets, masks)
            total_loss += loss.item()

            # Process predictions and ground truth for mAP calculation
            batch_pred_boxes = process_predictions(outputs, iou_threshold)
            batch_true_boxes = process_targets(targets)

            all_pred_boxes.extend(batch_pred_boxes)
            all_true_boxes.extend(batch_true_boxes)

    # Calculate average loss
    avg_loss = total_loss / len(test_loader)

    # Calculate mAP
    avg_map = calculate_map(all_pred_boxes, all_true_boxes, iou_threshold)

    print(f"Test Loss: {avg_loss:.4f}, mAP: {avg_map:.4f}")
    return avg_loss, avg_map


def train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    """
    Train the model and evaluate using mAP.
    """
    logging.info("Starting training...")
    prev_test_loss = float('inf')
    overfit_counter = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")

        for images, targets, masks in progress_bar:
            targets = [t.to(device) for t in targets] if targets else []
            masks = [m.to(device) for m in masks] if masks else []

            images = images.to(device)  # Images en batch
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})

        avg_train_loss = running_loss / len(train_loader)
        avg_test_loss, avg_map = evaluate_model(model, test_loader, criterion, device)

        logging.info(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}, mAP: {avg_map:.4f}")

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


def collate_fn(batch):
    """
    Custom collate function to handle variable number of bounding boxes.
    Args:
        batch: List of samples where each sample is (image, bboxes, mask).

    Returns:
        images: Tensor of images.
        bboxes: List of tensors containing bboxes.
        masks: List of tensors containing masks.
    """
    images = []
    bboxes = []
    masks = []

    for sample in batch:
        img, box, mask = sample
        images.append(img)  # Tensor of the image
        bboxes.append(box)  # Bounding boxes for the image
        masks.append(mask)  # Mask for the bounding boxes

    images = torch.stack(images, dim=0)  # Stack images into a batch tensor
    return images, bboxes, masks


if __name__ == "__main__":
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Dataset
    train_dataset = YoloDataset(
        data_yaml="/Users/baptiste/Documents/Developpement/Aix Ynov Campus Sas/Head-Dectection-PoC/head-detection-v1-1/data.yaml", 
        img_size=640, 
        mode="train"
    )

    test_dataset = YoloDataset(
        data_yaml="/Users/baptiste/Documents/Developpement/Aix Ynov Campus Sas/Head-Dectection-PoC/head-detection-v1-1/data.yaml", 
        img_size=640, 
        mode="test"
    )

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=collate_fn)

    # Model, optimizer, and loss function
    model = BBoxDetectionModel(num_bboxes=30).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = bbox_loss

    # Train the model
    train_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10)

    # Save the model
    torch.save(model.state_dict(), "bbox_detection_model.pth")
    logging.info("Model saved as bbox_detection_model.pth")
