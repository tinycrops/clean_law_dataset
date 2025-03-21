import os
import re
import pandas as pd
import glob
from bs4 import BeautifulSoup
from tqdm import tqdm
import argparse
import traceback
import signal
import time
import sys

# Set up a timeout handler
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Processing timed out")

def extract_info_from_html(file_path, timeout=60, debug=False):
    """
    Extract legal information from an ecode360 HTML file with timeout.
    
    Args:
        file_path (str): Path to the HTML file
        timeout (int): Timeout in seconds for processing a file
        debug (bool): Whether to print debug information
        
    Returns:
        list: List of dictionaries containing extracted information
    """
    # Set up the timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    try:
        if debug:
            print(f"Reading file: {file_path}")
        
        # Get file size
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
        if debug:
            print(f"File size: {file_size:.2f} MB")
        
        # For very large files, use a different approach
        if file_size > 10:  # If file is larger than 10MB
            print(f"Warning: Large file detected ({file_size:.2f} MB). Using chunk reading.")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = ""
                for i, chunk in enumerate(iter(lambda: file.read(1024*1024), "")):
                    if i < 10:  # Only read first 10MB for very large files
                        content += chunk
                    else:
                        print(f"File too large, only processing first 10MB of {file_path}")
                        break
        else:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
            except UnicodeDecodeError:
                if debug:
                    print(f"UTF-8 decoding failed, trying latin-1 for {file_path}")
                try:
                    with open(file_path, 'r', encoding='latin-1') as file:
                        content = file.read()
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
                    return []
        
        if debug:
            print(f"Successfully read file content, length: {len(content)} characters")
        
        # Get city name from filename
        city_name = os.path.basename(file_path).split('.')[0]
        if "City of " in city_name:
            city_name = city_name.split("City of ")[1].strip()
        
        if debug:
            print(f"Parsing HTML with BeautifulSoup for {city_name}")
        
        # Parse with a more lenient parser for problematic HTML
        soup = BeautifulSoup(content, 'html.parser')
        
        if debug:
            print("HTML parsed successfully")
        
        # Extract date if available
        date_text = ""
        print_header = soup.find('h1', class_='printHeader')
        if print_header:
            date_div = print_header.find_all('div')
            if len(date_div) > 1:
                date_text = date_div[1].text.strip()
                if debug:
                    print(f"Found date: {date_text}")
        
        # Find all relevant sections
        sections = []
        
        # Find chapter titles
        if debug:
            print("Searching for chapter titles...")
        chapter_titles = soup.find_all('h2', class_='title chapterTitle')
        if debug:
            print(f"Found {len(chapter_titles)} chapter titles")
        
        if debug:
            print("Searching for article titles...")
        article_titles = soup.find_all('h2', class_='title articleTitle')
        if debug:
            print(f"Found {len(article_titles)} article titles")
        
        if debug:
            print("Searching for section titles...")
        section_titles = soup.find_all('h4', class_='title sectionTitle')
        if debug:
            print(f"Found {len(section_titles)} section titles")
        
        # Process chapters
        for chapter in chapter_titles:
            if debug:
                print(f"Processing chapter: {chapter.text.strip()}")
            
            chapter_id = chapter.get('id', '')
            chapter_title = chapter.text.strip()
            chapter_content_div = soup.find('div', {'id': chapter_id, 'class': 'chapter_content content'})
            
            if chapter_content_div:
                # Extract revision history
                revision_dates = []
                for para in chapter_content_div.find_all('div', class_='para'):
                    text = para.text.strip()
                    if text.startswith('REVISED'):
                        revision_dates.append(text.replace('REVISED', '').strip())
                
                year = ""
                if revision_dates:
                    # Get the most recent revision year
                    year = revision_dates[-1].split()[-1]
                    if debug:
                        print(f"Found revision year: {year}")
            
                sections.append({
                    'Title': chapter_title.replace('Title', '').strip(),
                    'Section': chapter_title,
                    'Content': chapter_content_div.text.strip(),
                    'City': city_name,
                    'Year': year,
                    'Public Law Number': '',
                    'Statute Number': chapter_id,
                    'Bill enacted': '',
                    'Source': file_path,
                    'Date': date_text
                })
        
        # Process articles
        for article in article_titles:
            if debug:
                print(f"Processing article: {article.text.strip()}")
            
            article_id = article.get('id', '')
            article_title = article.text.strip()
            article_content_div = soup.find('div', {'id': article_id, 'class': 'article_content content'})
            
            sections.append({
                'Title': article_title.replace('Article', '').strip(),
                'Section': article_title,
                'Content': article_content_div.text.strip() if article_content_div else '',
                'City': city_name,
                'Year': '',  # Usually not specified at article level
                'Public Law Number': '',
                'Statute Number': article_id,
                'Bill enacted': '',
                'Source': file_path,
                'Date': date_text
            })
        
        # Process individual sections
        for section in section_titles:
            if debug and len(section_titles) > 100:
                # Only print debug for every 20th section if there are many
                if section_titles.index(section) % 20 == 0:
                    print(f"Processing section {section_titles.index(section)}/{len(section_titles)}")
            
            section_id = section.get('id', '')
            section_title = section.text.strip()
            section_content_div = soup.find('div', {'id': section_id, 'class': 'section_content content'})
            
            sections.append({
                'Title': '',  # Usually empty at section level
                'Section': section_title,
                'Content': section_content_div.text.strip() if section_content_div else '',
                'City': city_name,
                'Year': '',  # Usually not specified at section level
                'Public Law Number': '',
                'Statute Number': section_id,
                'Bill enacted': '',
                'Source': file_path,
                'Date': date_text
            })
        
        if debug:
            print(f"Finished processing file, extracted {len(sections)} sections")
        
        # Reset the alarm
        signal.alarm(0)
        return sections
    
    except TimeoutException:
        print(f"⚠️ Processing timed out for {file_path}")
        return [{
            'Title': 'PROCESSING_ERROR',
            'Section': 'Timeout',
            'Content': f'Processing timed out after {timeout} seconds',
            'City': os.path.basename(file_path),
            'Year': '',
            'Public Law Number': '',
            'Statute Number': '',
            'Bill enacted': '',
            'Source': file_path,
            'Date': ''
        }]
    except Exception as e:
        print(f"❌ Error processing {file_path}: {str(e)}")
        if debug:
            traceback.print_exc()
        return [{
            'Title': 'PROCESSING_ERROR',
            'Section': str(e),
            'Content': traceback.format_exc(),
            'City': os.path.basename(file_path),
            'Year': '',
            'Public Law Number': '',
            'Statute Number': '',
            'Bill enacted': '',
            'Source': file_path,
            'Date': ''
        }]
    finally:
        # Reset the alarm in case of any exception
        signal.alarm(0)

def process_html_files(directory, output_file, timeout=60, debug=False, batch_size=20):
    """
    Process HTML files in the directory and save results in batches.
    
    Args:
        directory (str): Directory containing HTML files
        output_file (str): Path to output parquet file
        timeout (int): Timeout in seconds for processing each file
        debug (bool): Whether to print debug information
        batch_size (int): Number of files to process before saving intermediate results
    """
    all_files = glob.glob(os.path.join(directory, "*.html"))
    print(f"Found {len(all_files)} HTML files to process")
    
    # Create a base name for intermediate files
    base_name = os.path.splitext(output_file)[0]
    
    # Process files in batches
    all_sections = []
    batch_num = 1
    
    for i, file_path in enumerate(tqdm(all_files, desc="Processing HTML files")):
        try:
            if debug:
                print(f"\n{'='*50}")
                print(f"Processing file {i+1}/{len(all_files)}: {file_path}")
            
            # Process the file with a timeout
            sections = extract_info_from_html(file_path, timeout=timeout, debug=debug)
            all_sections.extend(sections)
            
            # Save intermediate results after each batch
            if (i + 1) % batch_size == 0 or i == len(all_files) - 1:
                if all_sections:
                    df = pd.DataFrame(all_sections)
                    
                    # Save intermediate results
                    intermediate_file = f"{base_name}_batch{batch_num}.parquet"
                    df.to_parquet(intermediate_file, index=False)
                    print(f"Saved intermediate batch {batch_num} with {len(df)} sections to {intermediate_file}")
                    
                    # Reset for next batch
                    batch_num += 1
                    
                    # Keep the sections for the final combined file
                    # but clear memory if needed
                    if i < len(all_files) - 1:  # Not the last file
                        if debug:
                            current_memory = df.memory_usage(deep=True).sum() / (1024 * 1024)
                            print(f"Current DataFrame memory usage: {current_memory:.2f} MB")
                        
                        # If memory usage is high, clear the list to save memory
                        if df.shape[0] > 10000:
                            print("Large dataset detected, clearing memory...")
                            all_sections = []
        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            if debug:
                traceback.print_exc()
    
    # Combine all intermediate batches
    try:
        print("Combining all batches into final output...")
        all_batches = glob.glob(f"{base_name}_batch*.parquet")
        
        if all_batches:
            dfs = []
            for batch_file in tqdm(all_batches, desc="Loading batches"):
                try:
                    df_batch = pd.read_parquet(batch_file)
                    dfs.append(df_batch)
                except Exception as e:
                    print(f"Error reading batch {batch_file}: {str(e)}")
            
            if dfs:
                final_df = pd.concat(dfs, ignore_index=True)
                print(f"Combined {len(dfs)} batches, total {len(final_df)} sections")
                
                # Save final output
                final_df.to_parquet(output_file, index=False)
                print(f"Data saved to {output_file}")
                
                # Also save a CSV for easier inspection (only if not too large)
                if len(final_df) < 100000:  # Only save CSV if not too large
                    csv_output = output_file.replace('.parquet', '.csv')
                    final_df.to_csv(csv_output, index=False)
                    print(f"Data also saved to {csv_output} for easier inspection")
                else:
                    print(f"Dataset too large ({len(final_df)} rows) to save as CSV")
                
                # Cleanup intermediate files
                if not debug:
                    for batch_file in all_batches:
                        try:
                            os.remove(batch_file)
                        except:
                            pass
                    print("Cleaned up intermediate batch files")
                else:
                    print("Debug mode: keeping intermediate batch files")
            else:
                print("No valid batches found")
        else:
            print("No intermediate batches found")
    except Exception as e:
        print(f"Error combining batches: {str(e)}")
        if debug:
            traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract legal information from ecode360 HTML files")
    parser.add_argument("--input", "-i", required=True, help="Directory containing HTML files")
    parser.add_argument("--output", "-o", default="legal_data.parquet", help="Output parquet file path")
    parser.add_argument("--timeout", "-t", type=int, default=60, help="Timeout in seconds for processing each file")
    parser.add_argument("--batch", "-b", type=int, default=20, help="Number of files to process before saving intermediate results")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode with verbose output")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Input directory {args.input} does not exist")
    else:
        print(f"Starting processing with options:")
        print(f"- Input directory: {args.input}")
        print(f"- Output file: {args.output}")
        print(f"- Timeout: {args.timeout} seconds per file")
        print(f"- Batch size: {args.batch} files")
        print(f"- Debug mode: {'Enabled' if args.debug else 'Disabled'}")
        
        start_time = time.time()
        process_html_files(args.input, args.output, 
                          timeout=args.timeout, 
                          debug=args.debug,
                          batch_size=args.batch)
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds")
