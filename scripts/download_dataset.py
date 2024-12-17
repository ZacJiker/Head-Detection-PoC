import argparse
from roboflow import Roboflow

def download_dataset(version_number):
    rf = Roboflow(api_key="e6Dh3ZD6uI6eGuCaAajs")
    project = rf.workspace("insight-wpiwn").project("head-detection-v1-92vg2")
    
    # Accéder à la version demandée
    version = project.version(version_number)
    print(f"Downloading dataset for version {version_number}...")
    dataset = version.download("coco")  # Changez le format si nécessaire
    print("Download complete!")

def main():
    # Configuration de l'argument --version
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", required=True, help="The version number to download")
    args = parser.parse_args()
    
    download_dataset(args.version)  # Appel à la fonction avec la version

if __name__ == "__main__":
    main()
