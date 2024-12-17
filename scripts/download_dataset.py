import argparse
from roboflow import Roboflow
from typing import NoReturn
import sys

def download_dataset(version_number: int) -> NoReturn:
    """
    Downloads the specific dataset based on the version number.
    :param version_number: The version number of the dataset
    :raises: Exception if the download fails
    """
    try:
        rf = Roboflow(api_key="e6Dh3ZD6uI6eGuCaAajs")
        project = rf.workspace("insight-wpiwn").project("head-detection-v1-92vg2")

        # Access the requested version
        version = project.version(version_number)
        print(f"Downloading dataset for version {version_number}...")
        version.download("coco")  # Change the format if necessary
        print("Download complete!")
    except Exception as e:
        print(f"Error: {e}. Failed to download version {version_number}.")
        sys.exit(1)

def main() -> NoReturn:
    """
    Script entry point for CLI. Parses arguments and starts the download.
    """
    # CLI argument configuration
    parser = argparse.ArgumentParser(description="Download specific dataset version.")
    parser.add_argument(
        "--version",
        type=int,
        required=True,
        help="The version number of the dataset to download (e.g., 1, 2, 3)."
    )

    args = parser.parse_args()
    
    print("Starting dataset download...")
    download_dataset(args.version)

if __name__ == "__main__":
    main()
