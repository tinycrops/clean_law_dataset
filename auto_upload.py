import os
import argparse
import time
from datetime import datetime
from huggingface_hub import upload_folder, HfApi, create_repo
import pandas as pd

def get_total_citations(dataset_dir):
    """Get the total number of citations in the dataset"""
    csv_path = os.path.join(dataset_dir, "citations.csv")
    if not os.path.exists(csv_path):
        return 0
    try:
        df = pd.read_csv(csv_path)
        return len(df)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 0

def upload_dataset(dataset_dir, repo_id="tinycrops/american_law_citations", repo_type="dataset"):
    """Upload a dataset to the HuggingFace Hub
    
    Args:
        dataset_dir: Directory containing the dataset
        repo_id: HuggingFace repository ID
        repo_type: Repository type (dataset or model)
    """
    print(f"Uploading dataset from {dataset_dir} to {repo_id}...")
    
    # Create repo if it doesn't exist
    try:
        api = HfApi()
        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"Repository {repo_id} already exists")
        except Exception:
            print(f"Creating repository {repo_id}...")
            create_repo(repo_id=repo_id, repo_type=repo_type, exist_ok=True)
    except Exception as e:
        print(f"Error checking/creating repository: {e}")
    
    try:
        result = upload_folder(
            folder_path=dataset_dir,
            repo_id=repo_id,
            repo_type=repo_type
        )
        print(f"Upload complete. Repository URL: {result}")
        return result
    except Exception as e:
        print(f"Error uploading dataset: {e}")
        return None

def auto_upload(dataset_dir, repo_id, check_interval=600, min_new_citations=100):
    """Automatically upload the dataset as it grows
    
    Args:
        dataset_dir: Directory containing the dataset
        repo_id: HuggingFace repository ID
        check_interval: Interval in seconds between checks
        min_new_citations: Minimum number of new citations to trigger an upload
    """
    last_citation_count = get_total_citations(dataset_dir)
    last_upload_time = datetime.now()
    
    print(f"Starting auto-upload with {last_citation_count} initial citations")
    print(f"Will upload when at least {min_new_citations} new citations are found")
    
    while True:
        try:
            time.sleep(check_interval)
            current_time = datetime.now()
            current_citation_count = get_total_citations(dataset_dir)
            
            new_citations = current_citation_count - last_citation_count
            time_since_last_upload = (current_time - last_upload_time).total_seconds() / 60  # in minutes
            
            print(f"\nCheck at {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Current citations: {current_citation_count}")
            print(f"New citations since last upload: {new_citations}")
            print(f"Time since last upload: {time_since_last_upload:.1f} minutes")
            
            # Upload if enough new citations or if it's been a long time
            if new_citations >= min_new_citations or time_since_last_upload >= 30:
                print("Uploading dataset...")
                upload_dataset(dataset_dir, repo_id)
                last_citation_count = current_citation_count
                last_upload_time = current_time
            else:
                print(f"Not enough new citations yet. Waiting for {min_new_citations - new_citations} more")
            
        except KeyboardInterrupt:
            print("\nAuto-upload stopped by user.")
            break
        except Exception as e:
            print(f"Error in auto-upload: {e}")
            time.sleep(check_interval)  # Wait a bit before retrying

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-upload dataset to HuggingFace Hub")
    parser.add_argument("--dir", type=str, default="american_law_full", help="Directory containing the dataset")
    parser.add_argument("--repo", type=str, default="tinycrops/american_law_citations", help="HuggingFace repository ID")
    parser.add_argument("--interval", type=int, default=600, help="Check interval in seconds (default: 10 minutes)")
    parser.add_argument("--min-new", type=int, default=100, help="Minimum number of new citations to trigger an upload")
    
    args = parser.parse_args()
    auto_upload(args.dir, args.repo, args.interval, args.min_new) 