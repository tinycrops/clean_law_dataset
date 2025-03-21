from datasets import load_dataset
import json
import re

# Load the dataset
dataset = load_dataset("the-ride-never-ends/american_law", split="train", streaming=True)
dataset_iter = iter(dataset)

# Collect 20 samples to examine different document types
samples = []
for i in range(20):
    try:
        sample = next(dataset_iter)
        samples.append({
            'doc_id': sample['doc_id'],
            'doc_order': sample.get('doc_order', None),
            'html_title': sample.get('html_title', None)[:100] + '...' if sample.get('html_title') else None,
            'cid': sample.get('cid', None),
            'sample_html': sample.get('html', '')[:200] + '...' if sample.get('html') else None
        })
        print(f"Collected sample {i+1}: {sample['doc_id']}")
    except StopIteration:
        print("Reached end of dataset")
        break

# Save samples to a JSON file
with open('dataset_samples.json', 'w') as f:
    json.dump(samples, f, indent=2)

print(f"Saved {len(samples)} samples to dataset_samples.json")

# Now look for documents that might contain US Code citations (like "17 U.S.C. 501")
dataset_iter = iter(dataset)
citation_samples = []
usc_count = 0
cfr_count = 0
statute_count = 0

# Regular expressions for different citation formats
usc_pattern = re.compile(r'\d+\s+U\.?S\.?C\.?\s+[§\s]?\d+')
cfr_pattern = re.compile(r'\d+\s+C\.?F\.?R\.?\s+[§\s]?\d+')
statute_pattern = re.compile(r'(?:Public Law|P\.L\.|Stat\.)\s+\d+')

# Check documents
for i in range(5000):
    try:
        sample = next(dataset_iter)
        html_content = sample.get('html', '')
        
        citation_type = None
        match_text = None
        
        # Look for U.S.C. citations
        usc_match = usc_pattern.search(html_content)
        if usc_match and usc_count < 10:
            citation_type = "USC"
            match_text = usc_match.group(0)
            usc_count += 1
        
        # Look for C.F.R. citations
        if not citation_type:
            cfr_match = cfr_pattern.search(html_content)
            if cfr_match and cfr_count < 10:
                citation_type = "CFR"
                match_text = cfr_match.group(0)
                cfr_count += 1
        
        # Look for statute citations
        if not citation_type:
            statute_match = statute_pattern.search(html_content)
            if statute_match and statute_count < 10:
                citation_type = "Statute"
                match_text = statute_match.group(0)
                statute_count += 1
        
        if citation_type:
            # Get context around the match
            start_pos = max(0, html_content.find(match_text) - 100)
            end_pos = min(len(html_content), html_content.find(match_text) + len(match_text) + 100)
            context = html_content[start_pos:end_pos]
            
            citation_samples.append({
                'doc_id': sample['doc_id'],
                'cid': sample.get('cid', None),
                'citation_type': citation_type,
                'citation': match_text,
                'context': context
            })
            print(f"Found {citation_type} citation in document {i+1}: {match_text}")
            
            # Stop if we have enough samples
            if usc_count >= 10 and cfr_count >= 10 and statute_count >= 10:
                break
    except StopIteration:
        print("Reached end of dataset")
        break

# Save citation samples to a JSON file
with open('citation_samples.json', 'w') as f:
    json.dump(citation_samples, f, indent=2)

print(f"Saved {len(citation_samples)} citation samples to citation_samples.json")
print(f"USC citations: {usc_count}, CFR citations: {cfr_count}, Statute citations: {statute_count}") 