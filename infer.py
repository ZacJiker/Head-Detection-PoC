import torch
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from bbox_dataset import BBoxDataset
from train import BBoxDetectionModel

def plot_image_with_bboxes(image, bboxes, ax):
    """
    Plot a single image with bounding boxes.
    """
    image = image.permute(1, 2, 0).cpu().numpy()  # Convert image to HWC format for plotting
    image = np.clip(image, 0, 1)  # Ensure values are between 0 and 1
    ax.imshow(image)
    for bbox in bboxes:
        x_min, y_min, x_max, y_max = bbox
        rect = patches.Rectangle(
            (x_min, y_min), x_max - x_min, y_max - y_min,
            linewidth=2, edgecolor='r', facecolor='none'
        )
        ax.add_patch(rect)
    ax.axis("off")


def infer_bboxes(model, dataloader, device, threshold=0.5):
    """
    Perform inference and return bounding box predictions.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for images, _, _ in dataloader:
            images = images.to(device)
            outputs = model(images).cpu()
            for output in outputs:
                valid_bboxes = [bbox.tolist() for bbox in output if bbox[2] - bbox[0] > threshold]
                predictions.append(valid_bboxes)

    return predictions


def visualize_predictions(model, dataloader, device, num_images=10, threshold=0.5):
    """
    Visualize 10 random test images with predicted bounding boxes.
    """
    model.eval()
    sampled_indices = random.sample(range(len(dataloader.dataset)), num_images)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()

    with torch.no_grad():
        for i, idx in enumerate(sampled_indices):
            image, _, _ = dataloader.dataset[idx]
            image = image.unsqueeze(0).to(device)
            outputs = model(image).cpu().squeeze(0)
            
            valid_bboxes = [bbox.tolist() for bbox in outputs if bbox[2] - bbox[0] > threshold]
            plot_image_with_bboxes(image.squeeze(0), valid_bboxes, axes[i])
            axes[i].set_title(f"Image {idx}")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and dataloader
    dataset = BBoxDataset(
        annotations_file="path/to/_annotations.coco.json",
        images_dir="path/to/images",
        num_bboxes=100
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Load model
    model = BBoxDetectionModel(num_bboxes=100).to(device)
    model.load_state_dict(torch.load("bbox_detection_model.pth"))

    # Visualize predictions
    visualize_predictions(model, dataloader, device, num_images=10, threshold=0.5)
