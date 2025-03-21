import os
import re
import pandas as pd
import sys
import time
import traceback
from bs4 import BeautifulSoup
import glob
from tqdm import tqdm

def clean_text(text):
    """Clean text by removing extra whitespace and normalizing quotes"""
    if not text:
        return ""
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    # Replace non-breaking spaces with regular spaces
    text = text.replace('\xa0', ' ')
    return text.strip()

def clean_content(content_text):
    """Remove duplicate paragraphs and clean up content text"""
    if not content_text:
        return ""
    
    # Split into paragraphs
    paragraphs = content_text.split("\n")
    
    # Remove duplicate consecutive paragraphs
    unique_paragraphs = []
    for i, para in enumerate(paragraphs):
        if i == 0 or para != paragraphs[i-1]:
            unique_paragraphs.append(para)
    
    # Rejoin unique paragraphs
    return "\n".join(unique_paragraphs)

def extract_charter_document(file_path, output_file):
    """
    Extract legal information from a charter document HTML file with focus on proper structure
    
    Args:
        file_path (str): Path to the HTML file
        output_file (str): Path to save the output parquet file
    """
    print(f"Processing charter document: {file_path}")
    start_time = time.time()
    
    try:
        # Get city name from filename
        city_name = os.path.basename(file_path).split('.')[0]
        if "City of " in city_name:
            city_name = city_name.split("City of ")[1].strip()
        print(f"Extracting data for: {city_name}")
        
        # Read the file line by line to avoid memory issues
        print("Reading file...")
        
        # First, check the file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # For very large files, read in chunks
        if file_size_mb > 20:
            print("Large file detected, reading in chunks...")
            with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                content = ""
                chunk_size = 1024 * 1024  # 1MB chunks
                for i, chunk in enumerate(iter(lambda: file.read(chunk_size), "")):
                    content += chunk
                    if i > 30:  # Limit to ~30MB to avoid memory issues
                        print("File too large, processing first 30MB only")
                        break
        else:
            # For smaller files, read all at once
            try:
                with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
                    content = file.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                return False
        
        # Extract date
        date_match = re.search(r'<div>\s*([^<]*\d{4}[^<]*)\s*</div>', content)
        date_text = date_match.group(1).strip() if date_match else ""
        print(f"Document date: {date_text}")
        
        # Parse HTML with a more lenient parser
        print("Parsing HTML...")
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract all titles and sections
        print("Extracting sections...")
        
        # First, look for chapter titles (typically the Charter title)
        chapter_titles = soup.find_all('h2', class_='title chapterTitle')
        article_titles = soup.find_all('h2', class_='title articleTitle')
        section_titles = soup.find_all('h4', class_='title sectionTitle')
        
        print(f"Found {len(chapter_titles)} chapters, {len(article_titles)} articles, {len(section_titles)} sections")
        
        # Extract revision year if available
        revision_year = ""
        chapter_content = None
        
        for chapter in chapter_titles:
            chapter_id = chapter.get('id', '')
            chapter_content = soup.find('div', {'id': chapter_id, 'class': 'chapter_content content'})
            if chapter_content:
                break
        
        if chapter_content:
            # Look for revision dates
            revision_dates = []
            for para in chapter_content.find_all('div', class_='para'):
                text = para.text.strip()
                if "REVISED" in text:
                    year_match = re.search(r'REVISED\s+\w+\s+(\d{4})', text)
                    if year_match:
                        revision_dates.append(year_match.group(1))
            
            if revision_dates:
                revision_year = max(revision_dates)
                print(f"Found revision year: {revision_year}")
        
        # Process sections
        sections = []
        
        # Process chapters
        for chapter in chapter_titles:
            chapter_id = chapter.get('id', '')
            chapter_title = clean_text(chapter.text)
            chapter_content_div = soup.find('div', {'id': chapter_id, 'class': 'chapter_content content'})
            
            # Extract content if available
            content_text = ""
            if chapter_content_div:
                paragraphs = []
                for para in chapter_content_div.find_all('div', class_=re.compile(r'para|level|litem_content')):
                    para_text = clean_text(para.text)
                    if para_text and not para_text.startswith("REVISED"):
                        paragraphs.append(para_text)
                
                content_text = "\n".join(paragraphs)
            
            # Add to sections
            if content_text or not chapter_title.startswith('Title'):
                sections.append({
                    'Title': chapter_title.replace('Title ', ''),
                    'Section': chapter_title,
                    'Content': clean_content(content_text),
                    'City': city_name,
                    'Year': revision_year,
                    'Public Law Number': '',
                    'Statute Number': chapter_id,
                    'Bill enacted': '',
                    'Source': file_path,
                    'Date': date_text
                })
        
        # Process articles
        for article in article_titles:
            article_id = article.get('id', '')
            article_title = clean_text(article.text)
            article_content_div = soup.find('div', {'id': article_id, 'class': 'article_content content'})
            
            # Extract content if available
            content_text = ""
            if article_content_div:
                paragraphs = []
                for para in article_content_div.find_all('div', class_=re.compile(r'para|level|litem_content')):
                    para_text = clean_text(para.text)
                    if para_text:
                        paragraphs.append(para_text)
                
                content_text = "\n".join(paragraphs)
            
            # Add to sections
            sections.append({
                'Title': article_title.replace('ARTICLE ', '').replace('Article ', ''),
                'Section': article_title,
                'Content': clean_content(content_text),
                'City': city_name,
                'Year': revision_year,
                'Public Law Number': '',
                'Statute Number': article_id,
                'Bill enacted': '',
                'Source': file_path,
                'Date': date_text
            })
        
        # Process individual sections
        for section in section_titles:
            section_id = section.get('id', '')
            section_title = clean_text(section.text)
            section_content_div = soup.find('div', {'id': section_id, 'class': 'section_content content'})
            
            # Extract section number if available
            section_number = ""
            section_name = section_title
            
            # For formats like "Section 100. NAME."
            section_match = re.match(r'(?:§|Section)\s*(\d+)\.?\s*(.*)', section_title, re.IGNORECASE)
            if section_match:
                section_number = section_match.group(1).strip()
                section_name = section_match.group(2).strip()
            
            # Extract content if available
            content_text = ""
            if section_content_div:
                paragraphs = []
                for para in section_content_div.find_all(['div', 'p'], class_=re.compile(r'para|level|litem_content')):
                    para_text = clean_text(para.text)
                    if para_text:
                        paragraphs.append(para_text)
                
                content_text = "\n".join(paragraphs)
            
            # Add to sections
            sections.append({
                'Title': section_number + ". " + section_name if section_number else section_name,
                'Section': section_title,
                'Content': clean_content(content_text),
                'City': city_name,
                'Year': revision_year,
                'Public Law Number': '',
                'Statute Number': section_id,
                'Bill enacted': '',
                'Source': file_path,
                'Date': date_text
            })
        
        # Create DataFrame
        print(f"Extracted {len(sections)} sections")
        
        if sections:
            df = pd.DataFrame(sections)
            
            # Clean up the Title column
            df['Title'] = df['Title'].apply(lambda x: x.strip())
            df['Title'] = df['Title'].str.replace(r'^\.+\s*', '', regex=True)  # Remove leading periods
            
            # Save to parquet
            df.to_parquet(output_file, index=False)
            print(f"Data saved to {output_file}")
            
            # Also save a CSV for easier inspection
            csv_output = output_file.replace('.parquet', '.csv')
            df.to_csv(csv_output, index=False, encoding='utf-8')
            print(f"Data also saved to {csv_output} for easier inspection")
            
            return True
        else:
            print("No sections extracted")
            return False
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()
        return False
    finally:
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds")

def process_directory(directory, output_file):
    """Process all HTML files in a directory"""
    html_files = glob.glob(os.path.join(directory, "*.html"))
    print(f"Found {len(html_files)} HTML files")
    
    all_sections = []
    successful_files = 0
    
    for file_path in tqdm(html_files, desc="Processing files"):
        try:
            temp_output = file_path + ".temp.parquet"
            success = extract_charter_document(file_path, temp_output)
            
            # Read the results and add to the collection
            if success and os.path.exists(temp_output):
                df = pd.read_parquet(temp_output)
                all_sections.append(df)
                os.remove(temp_output)  # Clean up temp file
                successful_files += 1
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    print(f"Successfully processed {successful_files} out of {len(html_files)} files")
    
    if all_sections:
        # Combine all results
        combined_df = pd.concat(all_sections, ignore_index=True)
        
        # Save combined results
        combined_df.to_parquet(output_file, index=False)
        print(f"Combined data saved to {output_file}")
        
        # Also save a CSV
        csv_output = output_file.replace('.parquet', '.csv')
        combined_df.to_csv(csv_output, index=False, encoding='utf-8')
        print(f"Combined data also saved to {csv_output}")
    else:
        print("No sections found in any files")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python final-charter-extractor.py <input_file_or_directory> [output_file]")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else "charter_data.parquet"
    
    if not os.path.exists(input_path):
        print(f"Input path {input_path} does not exist")
        sys.exit(1)
    
    if os.path.isdir(input_path):
        process_directory(input_path, output_file)
    else:
        extract_charter_document(input_path, output_file)
