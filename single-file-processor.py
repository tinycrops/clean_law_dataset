import pandas as pd
import os
import glob
import argparse
import re
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("file_processor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("file_processor")

def is_blank_text(text):
    """
    Check if text is effectively blank (empty, whitespace, or just punctuation)
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

def generate_markdown_document(document_df, output_dir):
    """
    Generate a Markdown document from a document DataFrame
    """
    try:
        # Get doc_id and title
        doc_id = document_df['doc_id'].iloc[0]
        
        # Try to extract a title from clean_title fields
        title_row = document_df[~document_df['clean_title'].apply(is_blank_text)].head(1)
        title = title_row['clean_title'].iloc[0] if not title_row.empty else "Untitled Document"
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Create a valid filename from title and doc_id
        # Replace invalid characters with underscores
        valid_filename = re.sub(r'[^\w\-\.]', '_', f"{doc_id}_{title}")
        # Limit the length of the filename
        if len(valid_filename) > 200:
            valid_filename = valid_filename[:197] + "..."
        valid_filename = valid_filename + ".md"
        
        # Create the full output path
        output_path = os.path.join(output_dir, valid_filename)
        
        # Sort by doc_order
        document_df = document_df.sort_values('doc_order')
        
        # Create the markdown content
        markdown_lines = [f"# {title}\n"]
        markdown_lines.append(f"Document ID: {doc_id}\n")
        
        # Add sections
        for _, row in document_df.iterrows():
            section_title = row['clean_title']
            section_text = row['clean_text']
            
            # Add a section title if non-blank
            if not is_blank_text(section_title):
                markdown_lines.append(f"## {section_title}\n")
            
            # Add section content if non-blank
            if not is_blank_text(section_text):
                markdown_lines.append(f"{section_text}\n")
        
        # Write the file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(markdown_lines))
        
        return output_path
    except Exception as e:
        logger.error(f"Error generating markdown document for doc_id {document_df['doc_id'].iloc[0]}: {str(e)}")
        return None

def process_parquet_file(file_path, output_dir):
    """Process a single parquet file and extract documents"""
    try:
        logger.info(f"Processing {file_path}")
        
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        # Get unique document IDs in this file
        unique_doc_ids = df['doc_id'].unique()
        logger.info(f"Found {len(unique_doc_ids)} unique documents in {file_path}")
        
        # Process each document
        processed_count = 0
        for doc_id in unique_doc_ids:
            document_df = df[df['doc_id'] == doc_id]
            
            if document_df.empty:
                logger.warning(f"No data found for document ID {doc_id}")
                continue
            
            # Generate the markdown document
            output_path = generate_markdown_document(document_df, output_dir)
            
            if output_path:
                processed_count += 1
                if processed_count % 10 == 0:
                    logger.info(f"Generated {processed_count} documents so far...")
        
        logger.info(f"Generated {processed_count} documents from {file_path}")
        return processed_count
    except Exception as e:
        logger.error(f"Error processing {file_path}: {str(e)}")
        return 0

def main():
    parser = argparse.ArgumentParser(description="Process parquet files to extract legal documents")
    parser.add_argument("--input", type=str, required=True, help="Directory containing parquet files")
    parser.add_argument("--output", type=str, default="generated_documents", help="Output directory")
    
    args = parser.parse_args()
    
    # Make sure input directory exists
    if not os.path.isdir(args.input):
        logger.error(f"Input directory {args.input} does not exist")
        return
    
    # Get all parquet files
    parquet_files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if not parquet_files:
        logger.error(f"No parquet files found in {args.input}")
        return
    
    logger.info(f"Found {len(parquet_files)} parquet files")
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Process each file
    total_documents = 0
    for file_path in parquet_files:
        documents_in_file = process_parquet_file(file_path, args.output)
        total_documents += documents_in_file
    
    logger.info(f"Processing complete. Generated a total of {total_documents} documents.")

if __name__ == "__main__":
    main()
