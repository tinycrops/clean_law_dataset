from datasets import load_dataset
import re
import pandas as pd
import os
import json
from tqdm import tqdm

class CitationParser:
    def __init__(self):
        # Regex patterns for different types of citations
        self.usc_pattern = re.compile(r'(\d+)\s+U\.?S\.?C\.?\s+[§\s]?(\d+[a-z]*(?:\([a-z]+\))*)')
        self.cfr_pattern = re.compile(r'(\d+)\s+C\.?F\.?R\.?\s+[§\s]?(\d+(?:\.\d+)*)')
        self.statute_pattern = re.compile(r'(?:Public Law|P\.L\.)\s+(\d+)[-]?(\d*)')
        self.stat_pattern = re.compile(r'(\d+)\s+Stat\.?\s+(\d+)')
    
    def parse_usc_citation(self, text):
        """Parse US Code citations like '17 U.S.C. 501'"""
        matches = self.usc_pattern.findall(text)
        results = []
        for match in matches:
            if len(match) >= 2:
                title = match[0]
                section = match[1]
                results.append({
                    'citation_type': 'USC',
                    'title': title,
                    'section': section,
                    'sql_query': f"SELECT * FROM usc WHERE title = '{title}' AND section = '{section}'"
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
                results.append({
                    'citation_type': 'CFR',
                    'title': title,
                    'part': section.split('.')[0] if '.' in section else section,
                    'section': section,
                    'sql_query': f"SELECT * FROM cfr WHERE title = '{title}' AND part = '{section.split('.')[0] if '.' in section else section}'"
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
                results.append({
                    'citation_type': 'Public Law',
                    'congress': congress,
                    'law_number': law_number,
                    'sql_query': f"SELECT * FROM public_laws WHERE congress = '{congress}' AND law_number = '{law_number}'"
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
                    'sql_query': f"SELECT * FROM statutes WHERE volume = '{volume}' AND page = '{page}'"
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

def process_dataset(dataset_path=None, output_dir="citation_output", max_samples=1000):
    """Process the dataset to extract citations"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    parser = CitationParser()
    citation_data = []
    
    # Create the dataset
    dataset = load_dataset("the-ride-never-ends/american_law", split="train", streaming=True)
    dataset_iter = iter(dataset)
    
    print(f"Processing up to {max_samples} documents...")
    for i in tqdm(range(max_samples)):
        try:
            sample = next(dataset_iter)
            html_content = sample.get('html', '')
            
            # Skip empty content
            if not html_content:
                continue
            
            # Parse citations
            citations = parser.parse_all_citations(html_content)
            
            # If we found citations, add to the data
            if citations:
                for citation in citations:
                    citation_data.append({
                        'doc_id': sample['doc_id'],
                        'cid': sample.get('cid', ''),
                        'citation_type': citation['citation_type'],
                        'citation_details': citation,
                        'sql_query': citation['sql_query']
                    })
        except StopIteration:
            print("Reached end of dataset")
            break
    
    # Save results to CSV
    if citation_data:
        df = pd.DataFrame(citation_data)
        csv_path = os.path.join(output_dir, "citation_data.csv")
        df.to_csv(csv_path, index=False)
        print(f"Saved {len(citation_data)} citations to {csv_path}")
        
        # Save a JSON file with the same data for easy inspection
        json_path = os.path.join(output_dir, "citation_data.json")
        with open(json_path, 'w') as f:
            json.dump(citation_data, f, indent=2)
        print(f"Saved JSON version to {json_path}")
        
        # Count by citation type
        type_counts = df['citation_type'].value_counts().to_dict()
        print("Citation counts by type:")
        for citation_type, count in type_counts.items():
            print(f"  {citation_type}: {count}")
    else:
        print("No citations found")

if __name__ == "__main__":
    process_dataset(max_samples=2000) 