import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import time
import sys
import traceback

def extract_file_incremental(file_path, output_file):
    """
    Process a single HTML file incrementally to avoid memory issues and timeouts.
    
    Args:
        file_path (str): Path to the HTML file
        output_file (str): Path to save the output parquet file
    """
    print(f"Processing file: {file_path}")
    start_time = time.time()
    
    # Get city name from filename
    city_name = os.path.basename(file_path).split('.')[0]
    if "City of " in city_name:
        city_name = city_name.split("City of ")[1].strip()
    
    print(f"Extracting data for: {city_name}")
    
    try:
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Read the file in chunks to avoid memory issues with large files
        print("Reading file in chunks...")
        
        # First pass - just extract the date
        date_text = ""
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            for i, line in enumerate(file):
                if '<h1 class="printHeader">' in line:
                    # Read a few more lines to capture the date
                    date_lines = [line]
                    for _ in range(5):  # Read up to 5 more lines
                        try:
                            date_lines.append(next(file))
                        except StopIteration:
                            break
                    
                    # Extract date from these lines
                    date_html = ''.join(date_lines)
                    date_soup = BeautifulSoup(date_html, 'html.parser')
                    date_divs = date_soup.find_all('div')
                    if len(date_divs) > 1:
                        date_text = date_divs[1].text.strip()
                        print(f"Found date: {date_text}")
                        break
                
                # Don't read too many lines if we haven't found the date yet
                if i > 1000:
                    break
        
        # Second pass - process the file in smaller chunks
        sections = []
        chunk_size = 10000  # Lines per chunk
        
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            # Find all chapter, article, and section titles by pattern matching
            # instead of loading the entire file into memory
            print("Scanning for legal sections...")
            
            line_buffer = []
            current_section = None
            section_type = None
            section_id = None
            section_title = None
            section_content = []
            
            # Track our progress
            line_count = 0
            last_report = 0
            
            for line in file:
                line_count += 1
                
                # Report progress periodically
                if line_count - last_report >= 100000:
                    print(f"Processed {line_count:,} lines...")
                    last_report = line_count
                
                # Look for chapter titles
                chapter_match = re.search(r'<h2 class="title chapterTitle" id="([^"]+)">', line)
                if chapter_match:
                    # Save the previous section if there was one
                    if current_section:
                        content_text = ' '.join(section_content).strip()
                        
                        # Extract year from content for chapters
                        year = ""
                        if section_type == "chapter":
                            for content_line in section_content:
                                if "REVISED" in content_line:
                                    try:
                                        year_match = re.search(r'REVISED\s+\w+\s+(\d{4})', content_line)
                                        if year_match:
                                            year = year_match.group(1)
                                    except:
                                        pass
                        
                        sections.append({
                            'Title': section_title.replace(section_type.title(), '').strip() if section_title else '',
                            'Section': section_title,
                            'Content': content_text,
                            'City': city_name,
                            'Year': year,
                            'Public Law Number': '',
                            'Statute Number': section_id,
                            'Bill enacted': '',
                            'Source': file_path,
                            'Date': date_text
                        })
                    
                    # Start a new chapter section
                    section_id = chapter_match.group(1)
                    section_type = "chapter"
                    section_content = []
                    current_section = True
                    
                    # Get the title from the next line or so
                    line_buffer = [line]
                    title_found = False
                    
                    for _ in range(5):  # Look ahead up to 5 lines to find the title
                        try:
                            next_line = next(file)
                            line_count += 1
                            line_buffer.append(next_line)
                            
                            if '>' in next_line and '<' in next_line:
                                title_text = re.sub(r'<[^>]+>', '', ''.join(line_buffer))
                                section_title = title_text.strip()
                                title_found = True
                                break
                        except StopIteration:
                            break
                    
                    if not title_found:
                        section_title = "Unknown Chapter"
                    
                    continue
                
                # Look for article titles
                article_match = re.search(r'<h2 class="title articleTitle" id="([^"]+)">', line)
                if article_match:
                    # Save the previous section if there was one
                    if current_section:
                        content_text = ' '.join(section_content).strip()
                        
                        # Extract year (only if it's a chapter)
                        year = ""
                        if section_type == "chapter":
                            for content_line in section_content:
                                if "REVISED" in content_line:
                                    try:
                                        year_match = re.search(r'REVISED\s+\w+\s+(\d{4})', content_line)
                                        if year_match:
                                            year = year_match.group(1)
                                    except:
                                        pass
                        
                        sections.append({
                            'Title': section_title.replace(section_type.title(), '').strip() if section_title else '',
                            'Section': section_title,
                            'Content': content_text,
                            'City': city_name,
                            'Year': year,
                            'Public Law Number': '',
                            'Statute Number': section_id,
                            'Bill enacted': '',
                            'Source': file_path,
                            'Date': date_text
                        })
                    
                    # Start a new article section
                    section_id = article_match.group(1)
                    section_type = "article"
                    section_content = []
                    current_section = True
                    
                    # Get the title from this + next few lines
                    line_buffer = [line]
                    title_found = False
                    
                    for _ in range(5):  # Look ahead up to 5 lines
                        try:
                            next_line = next(file)
                            line_count += 1
                            line_buffer.append(next_line)
                            
                            if '>' in next_line and '<' in next_line:
                                title_text = re.sub(r'<[^>]+>', '', ''.join(line_buffer))
                                section_title = title_text.strip()
                                title_found = True
                                break
                        except StopIteration:
                            break
                    
                    if not title_found:
                        section_title = "Unknown Article"
                    
                    continue
                
                # Look for section titles
                section_match = re.search(r'<h4 class="title sectionTitle" id="([^"]+)">', line)
                if section_match:
                    # Save the previous section if there was one
                    if current_section:
                        content_text = ' '.join(section_content).strip()
                        
                        # Extract year (only if it's a chapter)
                        year = ""
                        if section_type == "chapter":
                            for content_line in section_content:
                                if "REVISED" in content_line:
                                    try:
                                        year_match = re.search(r'REVISED\s+\w+\s+(\d{4})', content_line)
                                        if year_match:
                                            year = year_match.group(1)
                                    except:
                                        pass
                        
                        sections.append({
                            'Title': section_title.replace(section_type.title(), '').strip() if section_title else '',
                            'Section': section_title,
                            'Content': content_text,
                            'City': city_name,
                            'Year': year,
                            'Public Law Number': '',
                            'Statute Number': section_id,
                            'Bill enacted': '',
                            'Source': file_path,
                            'Date': date_text
                        })
                    
                    # Start a new section
                    section_id = section_match.group(1)
                    section_type = "section"
                    section_content = []
                    current_section = True
                    
                    # Get the title from this + next few lines
                    line_buffer = [line]
                    title_found = False
                    
                    for _ in range(5):  # Look ahead up to 5 lines
                        try:
                            next_line = next(file)
                            line_count += 1
                            line_buffer.append(next_line)
                            
                            if '>' in next_line and '<' in next_line:
                                title_text = re.sub(r'<[^>]+>', '', ''.join(line_buffer))
                                section_title = title_text.strip()
                                title_found = True
                                break
                        except StopIteration:
                            break
                    
                    if not title_found:
                        section_title = "Unknown Section"
                    
                    continue
                
                # If we're inside a section, collect its content
                if current_section:
                    # Only collect content from relevant divs
                    if 'class="para"' in line or 'class="level"' in line or 'class="litem_content content"' in line:
                        # Clean the HTML tags
                        clean_line = re.sub(r'<[^>]+>', ' ', line).strip()
                        if clean_line:
                            section_content.append(clean_line)
            
            # Don't forget the last section
            if current_section:
                content_text = ' '.join(section_content).strip()
                
                # Extract year
                year = ""
                if section_type == "chapter":
                    for content_line in section_content:
                        if "REVISED" in content_line:
                            try:
                                year_match = re.search(r'REVISED\s+\w+\s+(\d{4})', content_line)
                                if year_match:
                                    year = year_match.group(1)
                            except:
                                pass
                
                sections.append({
                    'Title': section_title.replace(section_type.title(), '').strip() if section_title else '',
                    'Section': section_title,
                    'Content': content_text,
                    'City': city_name,
                    'Year': year,
                    'Public Law Number': '',
                    'Statute Number': section_id,
                    'Bill enacted': '',
                    'Source': file_path,
                    'Date': date_text
                })
        
        # Create DataFrame and save results
        print(f"Extracted {len(sections)} sections")
        
        if sections:
            df = pd.DataFrame(sections)
            
            # Save to parquet
            df.to_parquet(output_file, index=False)
            print(f"Data saved to {output_file}")
            
            # Also save a CSV for easier inspection
            csv_output = output_file.replace('.parquet', '.csv')
            df.to_csv(csv_output, index=False)
            print(f"Data also saved to {csv_output} for easier inspection")
        else:
            print("No data extracted from the file")
        
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python single-file-extractor.py <input_file> <output_file>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    if not os.path.exists(input_file):
        print(f"Input file {input_file} does not exist")
        sys.exit(1)
    
    extract_file_incremental(input_file, output_file)
