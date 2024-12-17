import cv2
import os
import roboflow
import random

from tqdm import tqdm

def extract_frames_and_upload_to_roboflow(
    video_path: str,
    output_dir: str,
    workspace_name: str,
    project_name: str,
    frame_interval: int = 1,
    test_split_ratio: float = 0.2
) -> None:
    """
    Extracts frames from a video and uploads them to a Roboflow project with train/test splitting.

    Parameters
    ----------
    video_path : str
        Path to the input video file.
    output_dir : str
        Directory to temporarily save extracted frames.
    workspace_name : str
        Name of the Roboflow workspace.
    project_name : str
        Name of the Roboflow project.
    frame_interval : int, optional
        Interval at which frames are extracted (default is 1, meaning every frame).
    test_split_ratio : float, optional
        Proportion of frames to be assigned to the "test" split (default is 0.2).

    Raises
    ------
    FileNotFoundError
        If the video file does not exist.
    ValueError
        If the video cannot be opened.
    """
    # Validate input arguments
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Specified video file does not exist: {video_path}")
    if not (0 <= test_split_ratio <= 1):
        raise ValueError("test_split_ratio must be between 0 and 1.")

    # Authenticate with Roboflow
    rf = roboflow.Roboflow(api_key="e6Dh3ZD6uI6eGuCaAajs")
    workspace = rf.workspace(workspace_name)
    project = workspace.project(project_name)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    if not video_capture.isOpened():
        raise ValueError(f"Unable to open the video file: {video_path}")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Retrieve total frame count for progress tracking
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise ValueError("Video file contains no frames.")

    # Initialize frame counters
    frame_index = 0
    saved_count = 0

    # Set up a progress bar
    with tqdm(total=total_frames // frame_interval, desc="Processing Frames", unit="frame") as pbar:
        while True:
            # Read a frame from the video
            success, frame = video_capture.read()
            if not success:
                break  # Exit loop when no more frames are available

            # Save frame if it matches the specified interval
            if frame_index % frame_interval == 0:
                output_path = os.path.join(output_dir, f"frame_{saved_count:04d}.jpg")
                cv2.imwrite(output_path, frame)

                # Assign to either the "train" or "test" split
                split = "test" if random.random() < test_split_ratio else "train"

                # Attempt to upload the frame to Roboflow
                try:
                    project.upload(image_path=output_path, split=split)
                    saved_count += 1
                except Exception as e:
                    print(f"Error uploading frame {saved_count:04d}: {e}")

                # Update progress bar
                pbar.update(1)

            frame_index += 1

    # Release video resources
    video_capture.release()

    # Final log message
    print(f"Processing complete: {saved_count} frames uploaded to Roboflow.")


# Example usage
if __name__ == "__main__":
    video_path = "/Users/baptiste/Downloads/Head-Dectection-PoC/IMG_2685.MOV"
    output_dir = "/Users/baptiste/Downloads/Head-Dectection-PoC/TempFrames"
    workspace_name = "insight-wpiwn"
    project_name = "head-detection-v1-92vg2"
    frame_interval = 1
    test_split_ratio = 0.2

    extract_frames_and_upload_to_roboflow(
        video_path=video_path,
        output_dir=output_dir,
        workspace_name=workspace_name,
        project_name=project_name,
        frame_interval=frame_interval,
        test_split_ratio=test_split_ratio
    )