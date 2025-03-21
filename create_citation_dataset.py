from datasets import load_dataset, Dataset
import pandas as pd
import numpy as np
import os
import json
import re
from tqdm import tqdm
from pathlib import Path

class CitationParser:
    def __init__(self):
        # Regex patterns for different types of citations - improved to catch more citation formats
        self.usc_pattern = re.compile(r'(\d+)\s+U\.?S\.?C\.?(?:ode)?(?:\s+[§\s]?|\s+section\s+|\s+sec\.\s+)(\d+[a-z]*(?:\([a-z0-9]+\))*)')
        self.cfr_pattern = re.compile(r'(\d+)\s+C\.?F\.?R\.?(?:\s+[§\s]?|\s+section\s+|\s+sec\.\s+)(\d+(?:\.\d+)*)')
        self.statute_pattern = re.compile(r'(?:Public\s+Law|P\.L\.)\s+(\d+)[-–—]?(\d*)')
        self.stat_pattern = re.compile(r'(\d+)\s+Stat\.?(?:utes)?(?:\s+at\s+Large)?\s+(\d+)')
    
    def parse_usc_citation(self, text):
        """Parse US Code citations like '17 U.S.C. 501'"""
        matches = self.usc_pattern.findall(text)
        results = []
        for match in matches:
            if len(match) >= 2:
                title = match[0]
                section = match[1]
                # Normalize section to handle subsections
                section_main = section.split('(')[0] if '(' in section else section
                
                results.append({
                    'citation_type': 'USC',
                    'title': title,
                    'section': section,
                    'section_main': section_main,
                    'sql_query': f"SELECT * FROM usc WHERE title = '{title}' AND section = '{section_main}'",
                    'full_citation': f"{title} U.S.C. {section}"
                })
        return results
    
    def parse_cfr_citation(self, text):
        """Parse Code of Federal Regulations citations like '40 CFR 261'"""
        matches = self.cfr_pattern.findall(text)
        results = []
        for match in matches:
            if len(match) >= 2:
                title = match[0]
                section = match[1]
                part = section.split('.')[0] if '.' in section else section
                
                results.append({
                    'citation_type': 'CFR',
                    'title': title,
                    'part': part,
                    'section': section,
                    'sql_query': f"SELECT * FROM cfr WHERE title = '{title}' AND part = '{part}'",
                    'full_citation': f"{title} C.F.R. {section}"
                })
        return results
    
    def parse_statute_citation(self, text):
        """Parse Public Law citations like 'Public Law 96-510' or 'P.L. 92-500'"""
        matches = self.statute_pattern.findall(text)
        results = []
        for match in matches:
            if len(match) >= 1:
                congress = match[0]
                law_number = match[1] if len(match) > 1 and match[1] else ""
                full_citation = f"Public Law {congress}" + (f"-{law_number}" if law_number else "")
                
                results.append({
                    'citation_type': 'Public Law',
                    'congress': congress,
                    'law_number': law_number,
                    'sql_query': f"SELECT * FROM public_laws WHERE congress = '{congress}' AND law_number = '{law_number}'",
                    'full_citation': full_citation
                })
        return results
    
    def parse_stat_citation(self, text):
        """Parse Statutes at Large citations like '124 Stat. 119'"""
        matches = self.stat_pattern.findall(text)
        results = []
        for match in matches:
            if len(match) >= 2:
                volume = match[0]
                page = match[1]
                
                results.append({
                    'citation_type': 'Stat',
                    'volume': volume,
                    'page': page,
                    'sql_query': f"SELECT * FROM statutes WHERE volume = '{volume}' AND page = '{page}'",
                    'full_citation': f"{volume} Stat. {page}"
                })
        return results
    
    def parse_all_citations(self, text):
        """Parse all types of citations in the text"""
        all_results = []
        all_results.extend(self.parse_usc_citation(text))
        all_results.extend(self.parse_cfr_citation(text))
        all_results.extend(self.parse_statute_citation(text))
        all_results.extend(self.parse_stat_citation(text))
        return all_results

def extract_relevant_text(html_content, citation_match, window_size=500):
    """Extract relevant text around the citation for context"""
    try:
        citation = citation_match['full_citation']
        # For the context, use a regex to find citation with some flexibility
        pattern = re.escape(citation).replace('\\ ', r'\s+')
        match = re.search(pattern, html_content)
        
        if match:
            # Increase window size for better context
            start_pos = max(0, match.start() - window_size)
            end_pos = min(len(html_content), match.end() + window_size)
            
            # Get context
            context = html_content[start_pos:end_pos]
            
            # If the context starts mid-sentence, try to find the beginning of the sentence
            if start_pos > 0 and not re.match(r'^\s*[A-Z]', context):
                # Look for the previous sentence boundary
                prev_text = html_content[max(0, start_pos - 200):start_pos]
                sentence_breaks = list(re.finditer(r'[.!?]\s+[A-Z]', prev_text))
                if sentence_breaks:
                    # Get the position of the last sentence break
                    last_break = sentence_breaks[-1].start() + 2  # +2 to include the space after the period
                    # Adjust the start position to include the complete sentence
                    new_start = max(0, start_pos - 200 + last_break)
                    context = html_content[new_start:end_pos]
            
            return context
        
        # If exact pattern not found, try a more general search
        title = citation_match.get('title', '') 
        if 'section' in citation_match:
            section = citation_match['section']
            # Use a more flexible pattern to find the citation
            pattern = f"{title}\\s+[UCS].+?{section}"
            match = re.search(pattern, html_content)
            if match:
                start_pos = max(0, match.start() - window_size)
                end_pos = min(len(html_content), match.end() + window_size)
                
                # Get context
                context = html_content[start_pos:end_pos]
                
                # If the context starts mid-sentence, try to find the beginning of the sentence
                if start_pos > 0 and not re.match(r'^\s*[A-Z]', context):
                    # Look for the previous sentence boundary
                    prev_text = html_content[max(0, start_pos - 200):start_pos]
                    sentence_breaks = list(re.finditer(r'[.!?]\s+[A-Z]', prev_text))
                    if sentence_breaks:
                        # Get the position of the last sentence break
                        last_break = sentence_breaks[-1].start() + 2  # +2 to include the space after the period
                        # Adjust the start position to include the complete sentence
                        new_start = max(0, start_pos - 200 + last_break)
                        context = html_content[new_start:end_pos]
                
                return context
            
        # If still not found, just return a general window
        return citation + " [Context not found]"
    except Exception as e:
        return citation + f" [Error extracting context: {str(e)}]"

def process_dataset(output_dir="american_law_full", max_samples=None, max_per_type=None, chunk_size=10000, resume_from=0):
    """Process the dataset to extract citations and create a new dataset
    
    Args:
        output_dir: Directory to save the dataset
        max_samples: Maximum number of documents to process (None for all)
        max_per_type: Maximum number of citations to collect per type (None for unlimited)
        chunk_size: Number of documents to process in each chunk before saving
        resume_from: Document index to resume processing from
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    parser = CitationParser()
    all_citations = []
    
    # Track citation counts by type
    citation_type_counts = {
        'USC': 0,
        'CFR': 0,
        'Public Law': 0,
        'Stat': 0
    }
    
    # Check if we're resuming from a previous run
    csv_path = os.path.join(output_dir, "citations.csv")
    if resume_from > 0 and os.path.exists(csv_path):
        print(f"Resuming from document {resume_from}, loading existing citations...")
        existing_df = pd.read_csv(csv_path)
        all_citations = existing_df.to_dict('records')
        
        # Update citation counts
        for citation in all_citations:
            citation_type = citation.get('citation_type')
            if citation_type in citation_type_counts:
                citation_type_counts[citation_type] += 1
        
        print(f"Loaded {len(all_citations)} existing citations")
        print(f"Current citation counts: {citation_type_counts}")
    
    # Load dataset
    print("Loading American Law dataset...")
    dataset = load_dataset("the-ride-never-ends/american_law", split="train", streaming=True)
    dataset_iter = iter(dataset)
    
    # Skip documents if resuming
    if resume_from > 0:
        print(f"Skipping {resume_from} documents...")
        for _ in tqdm(range(resume_from)):
            try:
                next(dataset_iter)
            except StopIteration:
                print("Reached end of dataset during skipping phase")
                break
        print(f"Skipped {resume_from} documents, resuming processing...")
    
    # Set total documents to process
    total_to_process = max_samples if max_samples is not None else 541790  # Total rows in the dataset
    print(f"Processing up to {total_to_process} documents to extract citations...")
    
    # Process in chunks
    docs_processed = resume_from
    chunk_start = docs_processed
    citations_in_current_chunk = 0
    chunk_num = (resume_from // chunk_size) + 1
    
    try:
        for i in tqdm(range(resume_from, total_to_process)):
            try:
                sample = next(dataset_iter)
                doc_id = sample.get('doc_id', '')
                doc_title = sample.get('html_title', '')
                html_content = sample.get('html', '')
                cid = sample.get('cid', '')
                
                # Skip empty content
                if not html_content:
                    continue
                
                # Parse citations
                citations = parser.parse_all_citations(html_content)
                
                # If we found citations, add to dataset
                if citations:
                    for citation in citations:
                        citation_type = citation['citation_type']
                        
                        # Skip if we have reached our limit for this type and max_per_type is set
                        if max_per_type and citation_type_counts.get(citation_type, 0) >= max_per_type:
                            continue
                        
                        # Get context around the citation
                        context = extract_relevant_text(html_content, citation)
                        
                        # Add additional metadata
                        citation_data = {
                            'doc_id': doc_id,
                            'doc_title': doc_title,
                            'cid': cid,
                            'citation_type': citation_type,
                            'full_citation': citation['full_citation'],
                            'context': context,
                            'sql_query': citation['sql_query']
                        }
                        
                        # Add type-specific fields
                        for key, value in citation.items():
                            if key not in citation_data:
                                citation_data[key] = value
                        
                        all_citations.append(citation_data)
                        citations_in_current_chunk += 1
                        
                        # Update citation type count
                        citation_type_counts[citation_type] = citation_type_counts.get(citation_type, 0) + 1
                
                docs_processed += 1
                
                # Print progress occasionally
                if docs_processed % 5000 == 0 or docs_processed == total_to_process:
                    total_citations = sum(citation_type_counts.values())
                    print(f"\nProcessed {docs_processed} documents")
                    print(f"Current citation counts: {citation_type_counts}")
                    print(f"Total citations: {total_citations}")
                
                # Save progress in chunks
                if docs_processed % chunk_size == 0 or docs_processed == total_to_process:
                    save_progress(all_citations, output_dir, citation_type_counts, chunk_num)
                    print(f"Saved chunk {chunk_num} with {citations_in_current_chunk} new citations. Total: {len(all_citations)}")
                    chunk_num += 1
                    chunk_start = docs_processed
                    citations_in_current_chunk = 0
            
            except StopIteration:
                print("Reached end of dataset")
                break
            except Exception as e:
                print(f"Error processing document {docs_processed}: {e}")
                # Continue with next document
                continue
        
        # Save final results if not already saved
        if citations_in_current_chunk > 0:
            save_progress(all_citations, output_dir, citation_type_counts, chunk_num)
    
    except KeyboardInterrupt:
        print("Processing interrupted. Saving current progress...")
        save_progress(all_citations, output_dir, citation_type_counts, chunk_num)
        print(f"Progress saved. Resume from document {chunk_start} to continue.")
        return
    
    print(f"Completed processing. Total documents processed: {docs_processed}")
    print(f"Total citations found: {len(all_citations)}")
    print(f"Citation counts by type: {citation_type_counts}")

def save_progress(all_citations, output_dir, citation_type_counts, chunk_num):
    """Save current progress to CSV and create dataset artifacts"""
    if not all_citations:
        print("No citations to save")
        return
    
    print(f"Saving progress... ({len(all_citations)} citations)")
    
    # Create a DataFrame with all citations
    df = pd.DataFrame(all_citations)
    
    # Check for specific nan values and replace them
    for col in df.columns:
        if col in df:
            df[col] = df[col].replace({np.nan: None})
    
    # Save as CSV for easy inspection and resuming
    csv_path = os.path.join(output_dir, "citations.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV version to {csv_path}")
    
    # Create a Hugging Face Dataset
    ds = Dataset.from_pandas(df)
    
    # Save as parquet files
    parquet_dir = os.path.join(output_dir, "data")
    if not os.path.exists(parquet_dir):
        os.makedirs(parquet_dir)
    
    print(f"Saving dataset to {parquet_dir}...")
    ds.save_to_disk(parquet_dir)
    
    # Count citation types
    type_counts = df['citation_type'].value_counts().to_dict()
    print("Citation counts by type:")
    for citation_type, count in type_counts.items():
        print(f"  {citation_type}: {count}")
    
    # Create dataset card
    dataset_card = f"""---
license: mit
datasets:
- the-ride-never-ends/american_law
language:
- en
size_categories:
- n>10K
---

# American Law Citations Dataset

This dataset contains {len(all_citations)} legal citations extracted from the American Law dataset.

## Dataset Description

The dataset contains parsed legal citations from various legal documents, including:

- U.S. Code (USC): {type_counts.get('USC', 0)} citations
- Code of Federal Regulations (CFR): {type_counts.get('CFR', 0)} citations
- Public Laws: {type_counts.get('Public Law', 0)} citations
- Statutes at Large: {type_counts.get('Stat', 0)} citations

Each citation includes the original document ID, document title, citation type, the full citation text, 
context surrounding the citation, and a SQL query that can be used to retrieve the cited document from a relational database.

## Dataset Creation

This dataset was created by parsing the text from the American Law dataset and extracting structured information 
about legal citations using regular expressions. The processing script identifies different types of legal citations 
and extracts relevant information such as title, section, and part numbers.

## Citation Types

1. **USC (United States Code)**: References to federal statutory law, e.g., "17 U.S.C. 501"
2. **CFR (Code of Federal Regulations)**: References to federal regulations, e.g., "40 CFR 261"
3. **Public Law**: References to laws as originally passed, e.g., "Public Law 89-655"
4. **Stat (Statutes at Large)**: References to the official publication of laws, e.g., "124 Stat. 119"

## Sample SQL Queries

The dataset includes SQL queries that can be used to retrieve the cited documents from a relational database:

- USC: `SELECT * FROM usc WHERE title = '17' AND section = '501'`
- CFR: `SELECT * FROM cfr WHERE title = '40' AND part = '261'`
- Public Law: `SELECT * FROM public_laws WHERE congress = '89' AND law_number = '655'`
- Stat: `SELECT * FROM statutes WHERE volume = '124' AND page = '119'`

## Usage

This dataset can be used for legal information retrieval, citation analysis, and building systems that link to primary legal sources.
"""
    
    # Save dataset card
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(dataset_card)
    
    # Create dataset info JSON
    dataset_info = {
        "description": "American Law Citations Dataset",
        "citation": "",
        "homepage": "https://huggingface.co/datasets/tinycrops/american_law_citations",
        "license": "mit",
        "features": {
            "doc_id": {"dtype": "string", "id": None, "_type": "Value"},
            "doc_title": {"dtype": "string", "id": None, "_type": "Value"},
            "cid": {"dtype": "string", "id": None, "_type": "Value"},
            "citation_type": {"dtype": "string", "id": None, "_type": "Value"},
            "full_citation": {"dtype": "string", "id": None, "_type": "Value"},
            "context": {"dtype": "string", "id": None, "_type": "Value"},
            "sql_query": {"dtype": "string", "id": None, "_type": "Value"}
        },
        "splits": {
            "data": {"name": "data", "num_bytes": None, "num_examples": len(all_citations), "dataset_name": "american_law_citations"}
        }
    }
    
    # Add type-specific features
    features = dataset_info["features"]
    if 'USC' in citation_type_counts and citation_type_counts['USC'] > 0:
        features["title"] = {"dtype": "string", "id": None, "_type": "Value"}
        features["section"] = {"dtype": "string", "id": None, "_type": "Value"}
        features["section_main"] = {"dtype": "string", "id": None, "_type": "Value"}
    
    if 'CFR' in citation_type_counts and citation_type_counts['CFR'] > 0:
        if "title" not in features:
            features["title"] = {"dtype": "string", "id": None, "_type": "Value"}
        features["part"] = {"dtype": "string", "id": None, "_type": "Value"}
        if "section" not in features:
            features["section"] = {"dtype": "string", "id": None, "_type": "Value"}
    
    if 'Public Law' in citation_type_counts and citation_type_counts['Public Law'] > 0:
        features["congress"] = {"dtype": "string", "id": None, "_type": "Value"}
        features["law_number"] = {"dtype": "string", "id": None, "_type": "Value"}
    
    if 'Stat' in citation_type_counts and citation_type_counts['Stat'] > 0:
        features["volume"] = {"dtype": "string", "id": None, "_type": "Value"}
        features["page"] = {"dtype": "string", "id": None, "_type": "Value"}
    
    # Save dataset info
    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print("Created dataset_info.json")

if __name__ == "__main__":
    process_dataset(output_dir="american_law_full", max_samples=None, max_per_type=None, chunk_size=10000, resume_from=0) 