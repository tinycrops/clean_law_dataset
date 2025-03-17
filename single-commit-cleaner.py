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

def clean_html_text(html_content):
    """
    Remove HTML tags and extract clean text from HTML content with robust error handling,
    while preserving structural elements like paragraphs, headings, and lists.
    """
    if not isinstance(html_content, str):
        return ""
    
    try:
        # Initial cleanup - decode HTML entities
        text = html.unescape(html_content)
        
        # Use BeautifulSoup to parse
        soup = BeautifulSoup(text, 'lxml')
        
        # Remove script and style elements
        for script_or_style in soup(["script", "style"]):
            script_or_style.decompose()
        
        # Handle different structural elements to preserve meaningful formatting
        # Replace paragraph, heading, div, and list item tags with newlines
        for tag in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'div', 'li']):
            tag.insert_before('\n')
            tag.insert_after('\n')
        
        # Replace breaks with newlines
        for br in soup.find_all('br'):
            br.replace_with('\n')
        
        # Get text - this preserves the newlines we just inserted
        clean_text = soup.get_text()
        
        # Normalize whitespace, but preserve paragraph structure
        # Replace multiple newlines with double newline to maintain paragraph breaks
        clean_text = re.sub(r'\n{2,}', '\n\n', clean_text)
        # Replace multiple spaces with single space
        clean_text = re.sub(r' +', ' ', clean_text)
        # Remove spaces at start/end of lines
        clean_text = re.sub(r'^ +| +$', '', clean_text, flags=re.MULTILINE)
        # Final trim
        clean_text = clean_text.strip()
        
        return clean_text
    except Exception as e:
        logger.warning(f"Error cleaning HTML content: {str(e)}")
        # Fall back to a more basic cleaning approach
        try:
            # More nuanced tag replacement for the fallback method
            # Insert newlines around common block elements
            text = re.sub(r'<(?:p|div|h\d|li)(?:\s[^>]*)?>', '\n', html_content, flags=re.IGNORECASE)
            text = re.sub(r'</(?:p|div|h\d|li)>', '\n', text, flags=re.IGNORECASE)
            # Replace br tags with newlines
            text = re.sub(r'<br\s*/?>|<br\s[^>]*>', '\n', text, flags=re.IGNORECASE)
            # Remove remaining tags
            text = re.sub(r'<[^>]+>', ' ', text)
            # Normalize whitespace while preserving paragraph structure
            text = re.sub(r'\n{2,}', '\n\n', text)
            text = re.sub(r' +', ' ', text)
            text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
            return text.strip()
        except Exception as e:
            logger.error(f"Failed to clean HTML content even with fallback method: {str(e)}")
            return ""

def process_batch(batch):
    """
    Process a single batch of data
    """
    try:
        # Create a DataFrame from the batch
        df = pd.DataFrame(batch)
        
        # Clean HTML content
        df['clean_text'] = df['html'].apply(clean_html_text)
        
        # Clean title content
        df['clean_title'] = df['html_title'].apply(clean_html_text)
        
        # Keep only necessary columns
        cleaned_df = df[['cid', 'doc_id', 'doc_order', 'clean_title', 'clean_text']]
        
        return cleaned_df
    except Exception as e:
        logger.error(f"Error processing batch: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def save_dataset_locally(dataset, output_dir, part_num):
    """
    Save a dataset to local storage in parquet format
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        file_path = os.path.join(output_dir, f"part-{part_num:05d}.parquet")
        dataset.to_parquet(file_path)
        return file_path
    except Exception as e:
        logger.error(f"Error saving dataset part {part_num}: {str(e)}")
        logger.error(traceback.format_exc())
        return None

def upload_entire_dataset(output_dir, repo_name, token):
    """
    Upload the entire dataset to Hugging Face Hub at once
    """
    try:
        # Create a Dataset object from all the parquet files
        logger.info(f"Loading all parquet files from {output_dir} for final upload")
        
        # Get all parquet files
        parquet_files = sorted(list(Path(output_dir).glob("*.parquet")))
        
        if not parquet_files:
            logger.error("No parquet files found to upload")
            return False
        
        logger.info(f"Found {len(parquet_files)} parquet files to upload")
        
        # Create a dataset from the files
        dataset = Dataset.from_parquet(parquet_files)
        
        # Upload to Hugging Face
        logger.info(f"Uploading final dataset to {repo_name}")
        
        # Retry mechanism for upload
        max_retries = 5
        retry_delay = 60  # Start with a longer delay due to rate limits
        
        for attempt in range(max_retries):
            try:
                dataset.push_to_hub(
                    repo_name,
                    private=False,
                    token=token,
                    max_shard_size="1GB"  # Use larger shards for the final upload
                )
                logger.info(f"Successfully uploaded entire dataset to {repo_name}")
                return True
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
                        logger.error("All upload attempts failed")
                        return False
        
        return False
    except Exception as e:
        logger.error(f"Error in final dataset upload: {str(e)}")
        logger.error(traceback.format_exc())
        return False

def process_dataset(batch_size=100, num_workers=4, output_dir="processed_data"):
    """
    Process the dataset in batches using parallel processing and save locally
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the dataset in streaming mode
    logger.info("Loading dataset (this may take a while for large datasets)...")
    try:
        dataset = load_dataset(
            "the-ride-never-ends/american_law", 
            split="train", 
            streaming=True
        )
        logger.info("Dataset loaded successfully!")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        logger.error(traceback.format_exc())
        return
    
    # Get Hugging Face token
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HF_TOKEN environment variable not set")
        return
    
    # Log in to Hugging Face
    try:
        login(token=hf_token)
        logger.info("Successfully logged in to Hugging Face")
    except Exception as e:
        logger.error(f"Failed to log in to Hugging Face: {str(e)}")
        return
    
    # Create the repository first if it doesn't exist
    try:
        api = HfApi()
        if not api.repo_exists(repo_id="tinycrops/clean_american_law", repo_type="dataset", token=hf_token):
            logger.info("Creating repository tinycrops/clean_american_law")
            api.create_repo(repo_id="tinycrops/clean_american_law", repo_type="dataset", private=False, token=hf_token)
    except Exception as e:
        logger.warning(f"Error checking/creating repository: {str(e)}")
        # Continue anyway, as it might already exist
    
    # Initialize counters
    processed_count = 0
    saved_count = 0
    error_count = 0
    part_num = 0
    
    # Start timing
    start_time = time.time()
    
    # Save metadata file to track progress
    def update_metadata():
        metadata = {
            "processed_batches": processed_count,
            "saved_parts": saved_count,
            "errors": error_count,
            "last_part": part_num,
            "timestamp": time.time()
        }
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
    
    # Check if we have a metadata file to resume from
    if os.path.exists(os.path.join(output_dir, "metadata.json")):
        try:
            with open(os.path.join(output_dir, "metadata.json"), "r") as f:
                metadata = json.load(f)
                processed_count = metadata.get("processed_batches", 0)
                saved_count = metadata.get("saved_parts", 0)
                error_count = metadata.get("errors", 0)
                part_num = metadata.get("last_part", 0)
                logger.info(f"Resuming from previous run. Processed {processed_count} batches, saved {saved_count} parts.")
        except Exception as e:
            logger.warning(f"Error loading metadata, starting fresh: {str(e)}")
    
    # Set up process pool
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Submit batches for processing
        futures = []
        
        logger.info(f"Starting dataset processing with batch size {batch_size} and {num_workers} workers")
        
        # Process batches with robust error handling
        dataset_iterator = dataset.iter(batch_size=batch_size)
        
        try:
            # Collect and process batches
            batch_count = 0
            processed_batches = []
            
            while True:
                try:
                    # Get a batch with retry logic
                    batch = None
                    retry_attempts = 0
                    max_retry = 5
                    
                    while batch is None and retry_attempts < max_retry:
                        try:
                            batch = next(dataset_iterator, None)
                            if batch is None:  # End of dataset
                                logger.info("Reached end of dataset")
                                break
                        except (pa.ArrowInvalid, pa.ArrowIOError) as arrow_err:
                            retry_attempts += 1
                            error_count += 1
                            logger.warning(f"Arrow error in batch: {str(arrow_err)}. Retry {retry_attempts}/{max_retry}")
                            time.sleep(1)  # Brief pause
                        except Exception as e:
                            retry_attempts += 1
                            error_count += 1
                            logger.warning(f"Error getting batch: {str(e)}. Retry {retry_attempts}/{max_retry}")
                            time.sleep(1)
                    
                    if batch is None:  # End of dataset or too many errors
                        break
                    
                    # Submit to executor
                    futures.append(executor.submit(process_batch, batch))
                    batch_count += 1
                    
                    # Log progress periodically
                    if batch_count % 100 == 0:
                        logger.info(f"Processed {batch_count} batches so far")
                        update_metadata()
                    
                    # If we have enough futures, start processing results
                    if len(futures) >= num_workers * 2:
                        # Process some completed futures
                        done_futures = [f for f in futures if f.done()]
                        for future in done_futures:
                            futures.remove(future)
                            try:
                                result = future.result()
                                if result is not None:
                                    # Convert to Dataset format
                                    batch_dataset = Dataset.from_pandas(result)
                                    
                                    # Append to list
                                    processed_batches.append(batch_dataset)
                                    processed_count += 1
                                    
                                    # Save locally periodically
                                    if len(processed_batches) >= 50:  # Save every 50 batches
                                        try:
                                            # Combine processed batches
                                            combined_dataset = concatenate_datasets(processed_batches)
                                            
                                            # Save to disk
                                            part_num += 1
                                            file_path = save_dataset_locally(combined_dataset, output_dir, part_num)
                                            
                                            if file_path:
                                                saved_count += 1
                                                logger.info(f"Saved part {part_num} to {file_path}")
                                            
                                            # Clear the list to free memory
                                            processed_batches = []
                                            
                                            # Force garbage collection
                                            gc.collect()
                                            
                                            # Update metadata
                                            update_metadata()
                                            
                                        except Exception as e:
                                            logger.error(f"Error combining or saving batches: {str(e)}")
                                            logger.error(traceback.format_exc())
                                            # Keep the processed batches and try again next time
                            except Exception as e:
                                logger.error(f"Error processing future result: {str(e)}")
                                logger.error(traceback.format_exc())
                
                except StopIteration:
                    logger.info("Reached end of dataset")
                    break
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error in main processing loop: {str(e)}")
                    logger.error(traceback.format_exc())
                    if error_count > 100:
                        logger.critical("Too many errors, aborting")
                        break
                    # Continue with the next batch
        
        except KeyboardInterrupt:
            logger.info("Process interrupted by user")
            # Save metadata before exiting
            update_metadata()
        
        # Process any remaining futures
        logger.info(f"Processing {len(futures)} remaining futures")
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing remaining batches"):
            try:
                result = future.result()
                if result is not None:
                    batch_dataset = Dataset.from_pandas(result)
                    processed_batches.append(batch_dataset)
                    processed_count += 1
            except Exception as e:
                logger.error(f"Error processing future result: {str(e)}")
    
    # Process any remaining batches
    if processed_batches:
        logger.info(f"Saving final {len(processed_batches)} processed batches")
        try:
            combined_dataset = concatenate_datasets(processed_batches)
            part_num += 1
            file_path = save_dataset_locally(combined_dataset, output_dir, part_num)
            
            if file_path:
                saved_count += 1
                logger.info(f"Saved final part {part_num} to {file_path}")
            
            # Update metadata
            update_metadata()
        except Exception as e:
            logger.error(f"Error saving final batches: {str(e)}")
            logger.error(traceback.format_exc())
    
    # Upload the entire dataset to Hugging Face
    logger.info("Processing complete. Starting final upload to Hugging Face...")
    upload_success = upload_entire_dataset(output_dir, "tinycrops/clean_american_law", hf_token)
    
    if upload_success:
        logger.info("Final upload to Hugging Face successful!")
    else:
        logger.error("Final upload to Hugging Face failed. You can retry manually later.")
        logger.info(f"The processed data is available in the {output_dir} directory")
    
    # Calculate and log processing statistics
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logger.info(f"Dataset processing completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    logger.info(f"Processed {processed_count} batches successfully out of {batch_count} total batches")
    logger.info(f"Saved {saved_count} parts locally")
    logger.info(f"Encountered {error_count} errors during processing")

def main():
    """
    Main function to execute the cleaning process
    """
    logger.info("Starting dataset cleaning process")
    
    # Make sure HF_TOKEN is set
    if not os.environ.get("HF_TOKEN"):
        logger.error("HF_TOKEN environment variable not set")
        logger.info("Please set your Hugging Face token with:")
        logger.info("export HF_TOKEN='your_token_here'")
        return
    
    # Process the dataset
    process_dataset(batch_size=100, num_workers=4)
    
    logger.info("Dataset cleaning completed")

if __name__ == "__main__":
    main()
