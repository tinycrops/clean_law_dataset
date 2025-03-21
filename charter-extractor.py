import os
import re
import pandas as pd
import sys
import time
import traceback
from bs4 import BeautifulSoup
import glob

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
        
        # Read the entire file at once - for charter documents this should be manageable
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            content = file.read()
        
        file_size_mb = len(content) / (1024 * 1024)
        print(f"File size: {file_size_mb:.2f} MB")
        
        # Extract date
        date_match = re.search(r'<div>\s*([^<]*\d{4}[^<]*)\s*</div>', content)
        date_text = date_match.group(1).strip() if date_match else ""
        print(f"Document date: {date_text}")
        
        # Parse HTML
        print("Parsing HTML...")
        soup = BeautifulSoup(content, 'html.parser')
        
        # First, check if we're dealing with a charter document
        is_charter = False
        charter_title = soup.find('h2', class_='title chapterTitle')
        if charter_title and ("Charter" in charter_title.text or "CHARTER" in charter_title.text):
            is_charter = True
            print("Charter document confirmed")
        
        # Extract sections in proper order with correct relationships
        sections = []
        
        # For tracking hierarchy
        current_article = None
        current_article_title = None
        revision_dates = []
        
        # Find the latest revision year
        chapter_content = soup.find('div', class_='chapter_content content')
        if chapter_content:
            for para in chapter_content.find_all('div', class_='para'):
                if para.text and "REVISED" in para.text:
                    revision_dates.append(para.text.strip())
        
        latest_year = ""
        if revision_dates:
            # Extract years from revision dates
            years = []
            for date in revision_dates:
                year_match = re.search(r'\b(19|20)\d{2}\b', date)
                if year_match:
                    years.append(year_match.group(0))
            
            if years:
                latest_year = max(years)
                print(f"Latest revision year: {latest_year}")
        
        # Process all sections in document order to maintain hierarchy
        all_titles = soup.find_all(['h2', 'h4'], class_=re.compile(r'title'))
        
        for title_elem in all_titles:
            elem_id = title_elem.get('id', '')
            title_text = clean_text(title_elem.text)
            
            # Skip empty titles
            if not title_text.strip():
                continue
            
            # Determine section type
            section_type = ""
            if 'chapterTitle' in title_elem.get('class', []):
                section_type = "Chapter"
            elif 'articleTitle' in title_elem.get('class', []):
                section_type = "Article"
                current_article = elem_id
                current_article_title = title_text
            elif 'sectionTitle' in title_elem.get('class', []):
                section_type = "Section"
            
            # Extract section number and name
            section_number = ""
            section_name = title_text
            
            # For sections like "§ 100. Name of the City."
            section_match = re.match(r'§\s*(\d+)\.?\s*(.*)', title_text)
            if section_match:
                section_number = section_match.group(1).strip()
                section_name = section_match.group(2).strip()
            
            # Get content div based on ID
            content_div = soup.find('div', {'id': elem_id, 'class': re.compile(r'.*_content content')})
            content_text = ""
            if content_div:
                # Get all paragraph text, preserving structure
                paras = []
                for para in content_div.find_all(['div', 'p'], class_=re.compile(r'para|litem_content')):
                    para_text = clean_text(para.text)
                    if para_text:
                        paras.append(para_text)
                
                content_text = "\n".join(paras)
            
            # Add to sections list
            bill_info = ""
            public_law = ""
            
            # Look for possible bill info in content
            if content_text and re.search(r'\b(Bill|Act)\b.*\b\d{4}\b', content_text, re.IGNORECASE):
                bill_match = re.search(r'\b(Bill|Act)\b.*\b\d{4}\b', content_text, re.IGNORECASE)
                if bill_match:
                    bill_info = bill_match.group(0)
            
            # Clean up section name
            if section_name.startswith("Title "):
                section_name = section_name.replace("Title ", "", 1)
            elif section_name.startswith("Article "):
                section_name = section_name.replace("Article ", "", 1)
            
            sections.append({
                'Title': section_name,
                'Section': title_text,
                'Content': content_text,
                'City': city_name,
                'Year': latest_year,
                'Public Law Number': public_law,
                'Statute Number': elem_id,
                'Bill enacted': bill_info,
                'Source': file_path,
                'Date': date_text,
                'Type': section_type,
                'Number': section_number,
                'Parent': current_article if section_type == "Section" else ""
            })
        
        # Create DataFrame and save results
        print(f"Extracted {len(sections)} sections")
        
        if sections:
            df = pd.DataFrame(sections)
            
            # Clean up the Title column for better organization
            # For chapters and articles, keep the name
            # For sections, use the article title + section name
            df['Title'] = df.apply(
                lambda row: f"{row['Title']}" if row['Type'] in ["Chapter", "Article"] else f"{row['Number']}. {row['Title']}", 
                axis=1
            )
            
            # Reorder columns to match the required format
            column_order = ['Title', 'Section', 'Content', 'City', 'Year', 'Public Law Number', 
                           'Statute Number', 'Bill enacted', 'Source', 'Date']
            
            # Keep only the columns requested in the output
            df = df[column_order]
            
            # Save to parquet
            df.to_parquet(output_file, index=False)
            print(f"Data saved to {output_file}")
            
            # Also save a CSV for easier inspection
            csv_output = output_file.replace('.parquet', '.csv')
            df.to_csv(csv_output, index=False, encoding='utf-8')
            print(f"Data also saved to {csv_output} for easier inspection")
        else:
            print("No data extracted from the file")
        
        elapsed_time = time.time() - start_time
        print(f"Processing completed in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        traceback.print_exc()

def process_directory(directory, output_file):
    """Process all HTML files in a directory"""
    html_files = glob.glob(os.path.join(directory, "*.html"))
    print(f"Found {len(html_files)} HTML files")
    
    all_sections = []
    
    for file_path in html_files:
        try:
            temp_output = file_path + ".temp.parquet"
            extract_charter_document(file_path, temp_output)
            
            # Read the results and add to the collection
            if os.path.exists(temp_output):
                df = pd.read_parquet(temp_output)
                all_sections.append(df)
                os.remove(temp_output)  # Clean up temp file
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
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
        print("Usage: python charter-extractor.py <input_file_or_directory> [output_file]")
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
