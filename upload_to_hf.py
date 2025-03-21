import os
import argparse
from huggingface_hub import upload_folder

def upload_dataset(dataset_dir, repo_id="tinycrops/american_law_citations", repo_type="dataset"):
    """Upload a dataset to the HuggingFace Hub
    
    Args:
        dataset_dir: Directory containing the dataset
        repo_id: HuggingFace repository ID
        repo_type: Repository type (dataset or model)
    """
    print(f"Uploading dataset from {dataset_dir} to {repo_id}...")
    result = upload_folder(
        folder_path=dataset_dir,
        repo_id=repo_id,
        repo_type=repo_type
    )
    print(f"Upload complete. Repository URL: {result}")
    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload dataset to HuggingFace Hub")
    parser.add_argument("--dir", type=str, default="american_law_full", help="Directory containing the dataset")
    parser.add_argument("--repo", type=str, default="tinycrops/american_law_citations", help="HuggingFace repository ID")
    
    args = parser.parse_args()
    upload_dataset(args.dir, args.repo) 