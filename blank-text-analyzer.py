import pandas as pd
import numpy as np
from datasets import load_dataset, Dataset
import os
from pathlib import Path
import logging
from tqdm import tqdm
import html
from bs4 import BeautifulSoup
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("blank_rows_analysis.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("blank_rows_analyzer")

def is_text_blank(text):
    """
    Check if text is effectively blank (empty, whitespace, or just punctuation).
    """
    if not isinstance(text, str):
        return True
    
    # Remove whitespace and check if empty
    stripped = text.strip()
    if not stripped:
        return True
    
    # Check if it's just punctuation and whitespace
    punctuation_pattern = re.compile(r'^[\s\.\,\;\:\!\?\(\)\[\]\{\}\-\_\=\+\*\/\\\|\'\"]+$')
    if punctuation_pattern.match(stripped):
        return True
    
    # Check if text is extremely short (e.g., just a single character)
    if len(stripped) <= 2:
        return True
    
    return False

def analyze_processed_files(processed_dir, sample_size=None):
    """
    Analyze the processed parquet files to identify blank rows.
    
    Args:
        processed_dir (str): Directory containing the processed parquet files
        sample_size (int, optional): Number of files to sample (for quick testing)
    
    Returns:
        tuple: (total_rows, blank_rows, blank_rows_data)
    """
    logger.info(f"Analyzing processed files in {processed_dir}")
    
    # Get all parquet files
    parquet_files = sorted(list(Path(processed_dir).glob("*.parquet")))
    
    if not parquet_files:
        logger.error(f"No parquet files found in {processed_dir}")
        return 0, 0, []
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Limit to sample size if specified
    if sample_size and sample_size < len(parquet_files):
        parquet_files = parquet_files[:sample_size]
        logger.info(f"Using sample of {sample_size} files")
    
    # Initialize counters and storage for blank rows
    total_rows = 0
    blank_rows = 0
    blank_rows_data = []
    
    # Process each file
    for file_path in tqdm(parquet_files, desc="Processing files"):
        try:
            # Read the parquet file
            df = pd.read_parquet(file_path)
            
            # Count total rows
            file_rows = len(df)
            total_rows += file_rows
            
            # Identify blank rows
            blank_mask = df['clean_text'].apply(is_text_blank)
            file_blank_rows = blank_mask.sum()
            blank_rows += file_blank_rows
            
            # If there are blank rows, save their IDs
            if file_blank_rows > 0:
                blank_df = df[blank_mask][['cid', 'doc_id', 'doc_order', 'clean_title', 'clean_text']]
                blank_rows_data.append(blank_df)
            
            # Log progress for large files
            if file_rows > 10000:
                logger.info(f"Processed {file_path.name}: {file_blank_rows} blank rows out of {file_rows}")
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
    
    # Combine all blank row data
    combined_blank_data = pd.concat(blank_rows_data) if blank_rows_data else pd.DataFrame()
    
    logger.info(f"Analysis complete. Found {blank_rows} blank rows out of {total_rows} total rows ({blank_rows/total_rows*100:.2f}%)")
    
    return total_rows, blank_rows, combined_blank_data

def fetch_original_html(blank_rows_data, batch_size=1000):
    """
    Fetch the original HTML content for blank rows from the source dataset.
    
    Args:
        blank_rows_data (DataFrame): DataFrame containing the blank rows
        batch_size (int): Batch size for processing the dataset
    
    Returns:
        DataFrame: DataFrame with blank rows and their original HTML content
    """
    logger.info("Fetching original HTML content for blank rows")
    
    if len(blank_rows_data) == 0:
        logger.info("No blank rows to fetch HTML for")
        return pd.DataFrame()
    
    # Create a dictionary of cid to row index for faster lookup
    blank_row_lookup = {row['cid']: i for i, row in blank_rows_data.iterrows()}
    
    # Create a new column for the original HTML
    blank_rows_data['original_html'] = ""
    
    try:
        # Load the original dataset in streaming mode
        logger.info("Loading original dataset (this may take a while)...")
        dataset = load_dataset(
            "the-ride-never-ends/american_law", 
            split="train", 
            streaming=True
        )
        
        # Process the original dataset in batches
        batch_count = 0
        matched_count = 0
        total_processed = 0
        
        for batch in tqdm(dataset.iter(batch_size=batch_size), desc="Searching original dataset"):
            batch_df = pd.DataFrame(batch)
            
            # Check if any of the rows in this batch match our blank rows
            matching_cids = set(batch_df['cid']) & set(blank_row_lookup.keys())
            
            if matching_cids:
                for cid in matching_cids:
                    # Get the original HTML for this CID
                    html_content = batch_df[batch_df['cid'] == cid]['html'].iloc[0]
                    
                    # Add it to our blank rows data
                    row_idx = blank_row_lookup[cid]
                    blank_rows_data.at[row_idx, 'original_html'] = html_content
                    
                    matched_count += 1
            
            batch_count += 1
            total_processed += len(batch_df)
            
            # Log progress
            if batch_count % 100 == 0:
                logger.info(f"Processed {batch_count} batches ({total_processed} rows), matched {matched_count}/{len(blank_rows_data)} blank rows")
            
            # If we've found all blank rows, we can stop
            if matched_count == len(blank_rows_data):
                logger.info("Found all blank rows, stopping search")
                break
        
        logger.info(f"Completed search. Matched {matched_count}/{len(blank_rows_data)} blank rows")
        
    except Exception as e:
        logger.error(f"Error fetching original HTML: {str(e)}")
    
    return blank_rows_data

def analyze_html_patterns(blank_rows_with_html):
    """
    Analyze patterns in the HTML content of blank rows.
    
    Args:
        blank_rows_with_html (DataFrame): DataFrame with blank rows and their HTML
    
    Returns:
        dict: Dictionary of pattern frequencies
    """
    logger.info("Analyzing HTML patterns in blank rows")
    
    if len(blank_rows_with_html) == 0:
        logger.info("No data to analyze patterns")
        return {}
    
    # Remove rows with missing HTML
    df = blank_rows_with_html[blank_rows_with_html['original_html'].notna()]
    
    if len(df) == 0:
        logger.info("No valid HTML data to analyze")
        return {}
    
    # Categorize the HTML patterns
    patterns = {}
    
    # Common empty patterns to check for
    empty_patterns = [
        ("<div class=\"chunk-content\"><br></div>", "Empty div with br"),
        ("<div class=\"chunk-content\"></div>", "Empty div"),
        ("<div class=\"chunk-content\">&nbsp;</div>", "Div with non-breaking space"),
        # Add more patterns as needed
    ]
    
    # Check for specific patterns
    for pattern, pattern_name in empty_patterns:
        pattern_count = df['original_html'].str.contains(pattern, regex=False).sum()
        patterns[pattern_name] = pattern_count
    
    # Check for patterns with minimal content
    has_img = df['original_html'].str.contains("<img", regex=False).sum()
    patterns["Contains image tags"] = has_img
    
    only_whitespace = 0
    has_table = df['original_html'].str.contains("<table", regex=False).sum()
    patterns["Contains table tags"] = has_table
    
    # Look for HTML with whitespace content
    for idx, row in df.iterrows():
        html_content = row['original_html']
        soup = BeautifulSoup(html_content, 'lxml')
        text_content = soup.get_text().strip()
        
        if not text_content:
            only_whitespace += 1
    
    patterns["Only whitespace after parsing"] = only_whitespace
    
    # Find other common patterns
    if len(df) > 100:
        # Sample random rows to examine
        sample = df.sample(min(100, len(df)))
        
        # Examine and categorize manually
        other_patterns = {}
        for _, row in sample.iterrows():
            html_content = row['original_html']
            # Look for specific structural patterns
            # This could be expanded based on what we find
            
            if "<div class=\"chunk-content\">" in html_content and "</div>" in html_content:
                content = re.search(r'<div class="chunk-content">(.*?)</div>', html_content, re.DOTALL)
                if content:
                    inner_content = content.group(1).strip()
                    if len(inner_content) < 50:  # Short content
                        key = f"Short content: {inner_content[:20]}..."
                        other_patterns[key] = other_patterns.get(key, 0) + 1
        
        # Add common patterns
        for pattern, count in sorted(other_patterns.items(), key=lambda x: x[1], reverse=True)[:10]:
            patterns[pattern] = count
    
    logger.info(f"Pattern analysis complete. Found {len(patterns)} distinct patterns")
    
    return patterns

def save_analysis_results(total_rows, blank_rows, blank_rows_with_html, patterns, output_dir="analysis_results"):
    """
    Save the analysis results to files.
    
    Args:
        total_rows (int): Total number of rows analyzed
        blank_rows (int): Number of blank rows found
        blank_rows_with_html (DataFrame): DataFrame with blank rows and their HTML
        patterns (dict): Dictionary of HTML patterns
        output_dir (str): Directory to save results to
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary
    summary = {
        "total_rows": total_rows,
        "blank_rows": blank_rows,
        "blank_percentage": round(blank_rows/total_rows*100, 2) if total_rows > 0 else 0,
        "patterns": patterns
    }
    
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(f"Analysis Summary\n")
        f.write(f"===============\n\n")
        f.write(f"Total rows analyzed: {total_rows:,}\n")
        f.write(f"Blank rows found: {blank_rows:,}\n")
        f.write(f"Blank percentage: {summary['blank_percentage']}%\n\n")
        f.write(f"Common HTML Patterns in Blank Rows:\n")
        f.write(f"==================================\n\n")
        
        for pattern, count in sorted(patterns.items(), key=lambda x: x[1], reverse=True):
            f.write(f"- {pattern}: {count:,} ({count/blank_rows*100:.2f}%)\n")
    
    # Save blank rows data
    if len(blank_rows_with_html) > 0:
        # Save a sample of the data (first 1000 rows)
        sample_size = min(1000, len(blank_rows_with_html))
        sample_df = blank_rows_with_html.head(sample_size)
        
        # Save to CSV
        sample_df.to_csv(os.path.join(output_dir, "blank_rows_sample.csv"), index=False)
        
        # Save to Excel for better viewing of HTML content
        try:
            sample_df.to_excel(os.path.join(output_dir, "blank_rows_sample.xlsx"), index=False)
        except Exception as e:
            logger.warning(f"Failed to save Excel file: {str(e)}")
        
        # Save full dataset to parquet
        blank_rows_with_html.to_parquet(os.path.join(output_dir, "all_blank_rows.parquet"))
    
    logger.info(f"Analysis results saved to {output_dir}")

def main():
    """
    Main function to execute the analysis process
    """
    logger.info("Starting blank rows analysis process")
    
    # Directory containing processed parquet files
    processed_dir = "processed_data"
    
    # Check if directory exists
    if not os.path.exists(processed_dir):
        logger.error(f"Directory {processed_dir} does not exist. Please provide the correct path.")
        return
    
    # Analyze the processed files
    total_rows, blank_rows, blank_rows_data = analyze_processed_files(processed_dir)
    
    if blank_rows == 0:
        logger.info("No blank rows found in the dataset")
        return
    
    # Fetch original HTML for blank rows
    blank_rows_with_html = fetch_original_html(blank_rows_data)
    
    # Analyze patterns in the HTML content
    patterns = analyze_html_patterns(blank_rows_with_html)
    
    # Save the analysis results
    save_analysis_results(total_rows, blank_rows, blank_rows_with_html, patterns)
    
    logger.info("Analysis completed")

if __name__ == "__main__":
    main()
