import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset, concatenate_datasets
from bs4 import BeautifulSoup
import re
import html
import os
import time
import logging
from tqdm import tqdm
from huggingface_hub import HfApi, login
import traceback
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow as pa
import json
from pathlib import Path
import glob

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("dataset_cleaning.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("dataset_cleaner")

def upload_entire_dataset(output_dir, repo_name, token):
    """
    Upload the entire dataset to Hugging Face Hub at once
    """
    try:
        # Get all parquet files as strings, not Path objects
        logger.info(f"Loading all parquet files from {output_dir} for final upload")
        parquet_files = sorted(glob.glob(os.path.join(output_dir, "*.parquet")))
        
        if not parquet_files:
            logger.error("No parquet files found to upload")
            return False
        
        logger.info(f"Found {len(parquet_files)} parquet files to upload")

        # Create a dataset from the files - chunk by chunk to avoid memory issues
        api = HfApi()
        
        # Create the repository if it doesn't exist
        if not api.repo_exists(repo_id=repo_name, repo_type="dataset", token=token):
            logger.info(f"Creating repository {repo_name}")
            api.create_repo(repo_id=repo_name, repo_type="dataset", private=False, token=token)
        
        logger.info(f"Uploading files to {repo_name}")
        
        # Upload chunk by chunk (10 files at a time)
        chunk_size = 10
        chunks = [parquet_files[i:i + chunk_size] for i in range(0, len(parquet_files), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} files)")
            
            # Create a temporary dataset from this chunk
            try:
                chunk_dataset = Dataset.from_parquet(chunk)
                
                # Retry mechanism for upload
                max_retries = 5
                retry_delay = 60  # Start with a longer delay due to rate limits
                
                for attempt in range(max_retries):
                    try:
                        # Use a different path for each chunk to avoid conflicts
                        chunk_path = f"{repo_name}_chunk{i+1}"
                        
                        chunk_dataset.push_to_hub(
                            chunk_path,
                            private=False,
                            token=token,
                            max_shard_size="500MB"
                        )
                        logger.info(f"Successfully uploaded chunk {i+1}")
                        break
                    except Exception as e:
                        if "Too Many Requests" in str(e) or "429" in str(e):
                            wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                            logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry.")
                            time.sleep(wait_time)
                        else:
                            logger.warning(f"Upload attempt {attempt+1} failed: {str(e)}")
                            if attempt < max_retries - 1:
                                logger.info(f"Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                                retry_delay *= 2
                            else:
                                logger.error(f"All upload attempts failed for chunk {i+1}")
                                return False
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {str(e)}")
                logger.error(traceback.format_exc())
                continue
        
        # Now create a dataset-dictionary.json file to combine all chunks
        logger.info("Creating dataset dictionary to combine all chunks")
        
        dataset_dict = {
            "version": "1.0.0",
            "splits": {
                "train": {
                    "chunks": [f"{repo_name}_chunk{i+1}" for i in range(len(chunks))]
                }
            }
        }
        
        # Write to a file
        dict_path = os.path.join(output_dir, "dataset-dictionary.json")
        with open(dict_path, "w") as f:
            json.dump(dataset_dict, f, indent=2)
        
        # Upload the dictionary
        try:
            api.upload_file(
                path_or_fileobj=dict_path,
                path_in_repo="dataset-dictionary.json",
                repo_id=repo_name,
                repo_type="dataset",
                token=token
            )
            logger.info("Successfully uploaded dataset dictionary")
            return True
        except Exception as e:
            logger.error(f"Error uploading dataset dictionary: {str(e)}")
            logger.error(traceback.format_exc())
            return False
        
    except Exception as e:
        logger.error(f"Error in final dataset upload: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Alternative simpler approach using the API directly
def upload_files_directly(output_dir, repo_name, token):
    """
    Upload files directly using the HfApi
    """
    try:
        logger.info(f"Uploading files directly from {output_dir} to {repo_name}")
        
        # Get all parquet files
        parquet_files = sorted(glob.glob(os.path.join(output_dir, "*.parquet")))
        
        if not parquet_files:
            logger.error("No parquet files found to upload")
            return False
        
        logger.info(f"Found {len(parquet_files)} parquet files to upload")
        
        # Create the repository if it doesn't exist
        api = HfApi()
        if not api.repo_exists(repo_id=repo_name, repo_type="dataset", token=token):
            logger.info(f"Creating repository {repo_name}")
            api.create_repo(repo_id=repo_name, repo_type="dataset", private=False, token=token)
        
        # Upload each file with retries
        for i, file_path in enumerate(parquet_files):
            max_retries = 5
            retry_delay = 60
            file_name = os.path.basename(file_path)
            logger.info(f"Uploading file {i+1}/{len(parquet_files)}: {file_name}")
            
            for attempt in range(max_retries):
                try:
                    api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=f"data/{file_name}",
                        repo_id=repo_name,
                        repo_type="dataset",
                        token=token
                    )
                    logger.info(f"Successfully uploaded {file_name}")
                    break
                except Exception as e:
                    if "Too Many Requests" in str(e) or "429" in str(e):
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"Rate limited. Waiting {wait_time} seconds before retry.")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"Upload attempt {attempt+1} failed for {file_name}: {str(e)}")
                        if attempt < max_retries - 1:
                            logger.info(f"Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            logger.error(f"All upload attempts failed for {file_name}")
                            # Continue with the next file
        
        # Create a metadata file
        metadata = {
            "description": "Clean version of the American Law dataset with HTML tags removed",
            "citation": "@dataset{american_law, author = {tinycrops}, title = {Clean American Law Dataset}}"
        }
        
        metadata_path = os.path.join(output_dir, "dataset_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Upload metadata
        try:
            api.upload_file(
                path_or_fileobj=metadata_path,
                path_in_repo="dataset_metadata.json",
                repo_id=repo_name,
                repo_type="dataset",
                token=token
            )
            logger.info("Successfully uploaded metadata")
        except Exception as e:
            logger.warning(f"Error uploading metadata: {str(e)}")
        
        return True
    
    except Exception as e:
        logger.error(f"Error in direct file upload: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def manual_upload_instructions(output_dir, repo_name):
    """
    Print instructions for manual upload
    """
    logger.info("\n" + "="*80)
    logger.info("MANUAL UPLOAD INSTRUCTIONS")
    logger.info("="*80)
    logger.info(f"To upload the dataset manually, use the following steps:")
    logger.info(f"1. Install the Hugging Face CLI: pip install -U huggingface_hub")
    logger.info(f"2. Log in to Hugging Face: huggingface-cli login")
    logger.info(f"3. Create a new dataset repository if it doesn't exist:")
    logger.info(f"   huggingface-cli repo create {repo_name} --type dataset")
    logger.info(f"4. Upload the processed files:")
    logger.info(f"   huggingface-cli upload {output_dir} {repo_name} --repo-type dataset")
    logger.info("="*80 + "\n")

# Modified main function to use the new approach
def main():
    """
    Main function for uploading
    """
    logger.info("Starting dataset upload")
    
    # Get Hugging Face token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set")
        logger.info("Please set your Hugging Face token with:")
        logger.info("export HF_TOKEN='your_token_here'")
        return
    
    # Make sure we have a directory to upload from
    output_dir = "processed_data"
    if not os.path.exists(output_dir) or not os.path.isdir(output_dir):
        logger.error(f"Directory {output_dir} does not exist")
        return
    
    # Try to log in to Hugging Face
    try:
        login(token=hf_token)
        logger.info("Successfully logged in to Hugging Face")
    except Exception as e:
        logger.error(f"Failed to log in to Hugging Face: {str(e)}")
        return
    
    # Try the direct file upload approach
    logger.info("Attempting direct file upload...")
    upload_success = upload_files_directly(output_dir, "tinycrops/clean_american_law", hf_token)
    
    if upload_success:
        logger.info("Direct file upload successful!")
    else:
        logger.error("Direct file upload failed.")
        manual_upload_instructions(output_dir, "tinycrops/clean_american_law")
    
    logger.info("Upload process completed")

if __name__ == "__main__":
    main()
