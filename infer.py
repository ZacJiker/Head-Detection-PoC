import torch

from bbox_dataset import BBoxDataset
from train import BBoxDetectionModel

def infer_bboxes(model, dataloader, device, threshold=0.5):
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


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = BBoxDataset(
        annotations_file="path/to/_annotations.coco.json",
        images_dir="path/to/images",
        num_bboxes=100
    )
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = BBoxDetectionModel(num_bboxes=100).to(device)
    model.load_state_dict(torch.load("bbox_detection_model.pth"))
    model.eval()

    predictions = infer_bboxes(model, dataloader, device)
    for i, pred in enumerate(predictions):
        print(f"Image {i}: {pred}")
