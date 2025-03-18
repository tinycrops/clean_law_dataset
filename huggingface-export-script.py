import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import logging
from tqdm import tqdm
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, login
import argparse
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("huggingface_export.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("huggingface_export")

class HuggingFaceExporter:
    """
    Export processed legal documents to Hugging Face dataset in a standardized format.
    """
    
    def __init__(self, input_dir, output_dir, username):
        """
        Initialize the exporter.
        
        Args:
            input_dir: Directory containing the processed legal documents
            output_dir: Directory to save the dataset artifacts
            username: Hugging Face username
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.username = username
        self.repo_name = f"{username}/structured_legal_documents"
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
    
    def load_document_index(self):
        """Load the document index JSON file."""
        index_path = os.path.join(self.input_dir, "document_index.json")
        
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Document index not found at {index_path}")
        
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def load_structured_documents(self):
        """Load the structured documents from the input directory."""
        structured_dir = os.path.join(self.input_dir, "structured")
        
        if not os.path.exists(structured_dir):
            raise FileNotFoundError(f"Structured documents directory not found at {structured_dir}")
        
        # Get all document directories
        document_dirs = [d for d in os.listdir(structured_dir) 
                        if os.path.isdir(os.path.join(structured_dir, d))]
        
        logger.info(f"Found {len(document_dirs)} root document directories")
        
        return document_dirs, structured_dir
    
    def extract_hierarchical_structure(self, doc_id, path, visited=None):
        """
        Extract the hierarchical structure of a document recursively.
        
        Args:
            doc_id: Document ID
            path: Path to the document directory
            visited: Set of already visited document IDs to avoid cycles
            
        Returns:
            Dict containing the document structure
        """
        if visited is None:
            visited = set()
            
        if doc_id in visited:
            return None  # Avoid cycles
            
        visited.add(doc_id)
        
        # Load document metadata
        metadata_path = os.path.join(path, "metadata.json")
        if not os.path.exists(metadata_path):
            return None
            
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        
        # Load document content
        doc_path = os.path.join(path, f"{doc_id}.md")
        content = ""
        if os.path.exists(doc_path):
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read()
        
        # Check for children directories
        children = []
        child_dirs = [d for d in os.listdir(path) 
                     if os.path.isdir(os.path.join(path, d)) and d != doc_id]
        
        for child_dir in child_dirs:
            child_path = os.path.join(path, child_dir)
            child_structure = self.extract_hierarchical_structure(
                child_dir, child_path, visited)
            if child_structure:
                children.append(child_structure)
        
        # Create document structure
        structure = {
            "id": doc_id,
            "title": metadata.get("title", ""),
            "chunks_count": metadata.get("chunks_count", 0),
            "content": content,
            "children": children
        }
        
        return structure
    
    def create_standardized_dataset(self, document_index, document_dirs, structured_dir):
        """
        Create a standardized dataset from the processed documents.
        
        Args:
            document_index: Document index JSON
            document_dirs: List of document directories
            structured_dir: Path to the structured documents directory
            
        Returns:
            DataFrame containing the standardized dataset
        """
        logger.info("Creating standardized dataset")
        
        records = []
        
        for doc_id in tqdm(document_dirs, desc="Processing documents"):
            doc_path = os.path.join(structured_dir, doc_id)
            
            # Extract hierarchical structure
            structure = self.extract_hierarchical_structure(doc_id, doc_path)
            
            if not structure:
                logger.warning(f"Could not extract structure for document {doc_id}")
                continue
                
            # Load flat version for full text
            flat_path = os.path.join(self.input_dir, "flat", f"{doc_id}.md")
            full_text = ""
            if os.path.exists(flat_path):
                with open(flat_path, "r", encoding="utf-8") as f:
                    full_text = f.read()
            
            # Create record
            record = {
                "doc_id": doc_id,
                "title": structure["title"],
                "full_text": full_text,
                "structure_json": json.dumps(structure),
                "num_chunks": structure["chunks_count"],
                "has_children": len(structure["children"]) > 0,
                "num_children": len(structure["children"]),
                "is_root": True  # These are all root documents
            }
            
            records.append(record)
            
            # Also add child documents to the dataset
            self._add_child_documents(structure, records)
        
        # Create dataframe
        df = pd.DataFrame(records)
        logger.info(f"Created dataset with {len(df)} documents")
        
        return df
    
    def _add_child_documents(self, parent_structure, records, depth=1):
        """Add child documents to the records list recursively."""
        for child in parent_structure["children"]:
            # Create record for child
            record = {
                "doc_id": child["id"],
                "title": child["title"],
                "full_text": child["content"],
                "structure_json": json.dumps(child),
                "num_chunks": child["chunks_count"],
                "has_children": len(child["children"]) > 0,
                "num_children": len(child["children"]),
                "is_root": False,
                "parent_id": parent_structure["id"],
                "depth": depth
            }
            
            records.append(record)
            
            # Process children recursively
            self._add_child_documents(child, records, depth + 1)
    
    def create_huggingface_dataset(self, df):
        """
        Create a Hugging Face dataset from the dataframe.
        
        Args:
            df: DataFrame containing the standardized dataset
            
        Returns:
            Dataset object
        """
        logger.info("Creating Hugging Face dataset")
        
        # Convert to Hugging Face dataset
        dataset = Dataset.from_pandas(df)
        
        # Create dataset dictionary
        dataset_dict = DatasetDict({
            "train": dataset
        })
        
        return dataset_dict
    
    def save_dataset_locally(self, dataset_dict):
        """
        Save the dataset locally.
        
        Args:
            dataset_dict: Dataset dictionary
            
        Returns:
            Path to the saved dataset
        """
        logger.info(f"Saving dataset locally to {self.output_dir}")
        
        # Save as parquet files
        dataset_dict.save_to_disk(self.output_dir)
        
        return self.output_dir
    
    def upload_to_huggingface(self, dataset_dict, token):
        """
        Upload the dataset to Hugging Face.
        
        Args:
            dataset_dict: Dataset dictionary
            token: Hugging Face API token
            
        Returns:
            Boolean indicating success
        """
        logger.info(f"Uploading dataset to Hugging Face as {self.repo_name}")
        
        try:
            # Log in to Hugging Face
            login(token=token, add_to_git_credential=True)
            
            # Push dataset to hub
            dataset_dict.push_to_hub(
                self.repo_name,
                token=token,
                private=False
            )
            
            logger.info(f"Successfully uploaded dataset to {self.repo_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading dataset to Hugging Face: {str(e)}")
            logger.error(traceback.format_exc())
            return False
    
    def create_readme(self):
        """Create a README.md file for the dataset."""
        readme_path = os.path.join(self.output_dir, "README.md")
        
        content = f"""# Structured Legal Documents Dataset

This dataset contains structured legal documents extracted from the American Law dataset. The documents are organized hierarchically, preserving their original structure.

## Dataset Structure

Each document has the following fields:

- `doc_id`: Unique identifier for the document
- `title`: Document title
- `full_text`: Complete text of the document, including all content chunks
- `structure_json`: JSON representation of the document's hierarchical structure
- `num_chunks`: Number of content chunks in the document
- `has_children`: Whether the document has child documents
- `num_children`: Number of child documents
- `is_root`: Whether the document is a root document (has no parent)
- `parent_id`: ID of the parent document (for non-root documents)
- `depth`: Depth in the document hierarchy (for non-root documents)

## Usage

```python
from datasets import load_dataset

# Load the dataset
dataset = load_dataset("{self.repo_name}")

# Access the documents
for doc in dataset["train"]:
    print(doc["title"])
    
    # Parse the structure
    import json
    structure_data = json.loads(doc["structure_json"])
    print(f"Document has {{len(structure_data['children'])}} children")
```

## License

This dataset is derived from the American Law dataset and is subject to the same license.

## Citation

```
@dataset{{structured_legal_documents,
  author = {{{self.username}}},
  title = {{Structured Legal Documents}},
  year = {{2025}},
}}
```
"""
        
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
            
        return readme_path
    
    def process(self, token=None):
        """
        Process the documents and create the Hugging Face dataset.
        
        Args:
            token: Hugging Face API token (optional)
            
        Returns:
            Boolean indicating success
        """
        try:
            # Load document index
            document_index = self.load_document_index()
            logger.info(f"Loaded document index with {len(document_index['documents'])} documents")
            
            # Load structured documents
            document_dirs, structured_dir = self.load_structured_documents()
            
            # Create standardized dataset
            df = self.create_standardized_dataset(document_index, document_dirs, structured_dir)
            
            # Save dataframe as CSV for reference
            csv_path = os.path.join(self.output_dir, "dataset.csv")
            df.to_csv(csv_path, index=False)
            logger.info(f"Saved dataset CSV to {csv_path}")
            
            # Create Hugging Face dataset
            dataset_dict = self.create_huggingface_dataset(df)
            
            # Save dataset locally
            local_path = self.save_dataset_locally(dataset_dict)
            
            # Create README
            readme_path = self.create_readme()
            logger.info(f"Created README at {readme_path}")
            
            # Upload to Hugging Face if token provided
            if token:
                success = self.upload_to_huggingface(dataset_dict, token)
                if success:
                    logger.info(f"Dataset available at: https://huggingface.co/datasets/{self.repo_name}")
                else:
                    logger.warning("Failed to upload dataset to Hugging Face")
            else:
                logger.info("No Hugging Face token provided, skipping upload")
                logger.info(f"Dataset saved locally at {local_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            logger.error(traceback.format_exc())
            return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Export legal documents to Hugging Face dataset")
    parser.add_argument("--input-dir", default="generated_documents", 
                        help="Directory containing processed legal documents")
    parser.add_argument("--output-dir", default="hf_dataset", 
                        help="Directory to save the dataset artifacts")
    parser.add_argument("--username", default="tinycrops", 
                        help="Hugging Face username")
    parser.add_argument("--token", default=None, 
                        help="Hugging Face API token (optional, also uses HF_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Get token from environment if not provided
    token = args.token or os.environ.get("HF_TOKEN")
    
    logger.info(f"Starting export process for {args.username}")
    
    exporter = HuggingFaceExporter(args.input_dir, args.output_dir, args.username)
    success = exporter.process(token)
    
    if success:
        logger.info("Export completed successfully")
    else:
        logger.error("Export failed")

if __name__ == "__main__":
    main()
