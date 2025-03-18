import pandas as pd
import numpy as np
from pathlib import Path
import os
import json
import logging
from tqdm import tqdm
from bs4 import BeautifulSoup
import re
import traceback
from typing import Dict, List, Tuple, Optional, Set
from datasets import Dataset, Features, Value, Sequence, DatasetDict
from huggingface_hub import HfApi, login

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("legal_dataset_exporter.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("legal_dataset_exporter")

class LegalDatasetExporter:
    """
    Export legal documents to a standardized Hugging Face dataset format
    """
    
    def __init__(self, input_dir: str, output_dir: str, repo_id: str = "tinycrops/structured_legal_documents"):
        """Initialize the exporter with input, output directories and repository ID."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.repo_id = repo_id
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics tracking
        self.stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "documents_with_children": 0,
            "max_chunks_per_document": 0,
            "max_hierarchy_depth": 0
        }
    
    def load_data(self, limit_files: Optional[int] = None) -> pd.DataFrame:
        """Load data from parquet files with optional file limit."""
        logger.info(f"Loading data from {self.input_dir}")
        
        parquet_files = sorted(list(Path(self.input_dir).glob("*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in {self.input_dir}")
        
        if limit_files:
            parquet_files = parquet_files[:limit_files]
        
        logger.info(f"Loading {len(parquet_files)} parquet files")
        
        dfs = []
        for file in tqdm(parquet_files, desc="Loading files"):
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {str(e)}")
                continue
        
        if not dfs:
            raise ValueError("No data loaded from parquet files")
        
        data = pd.concat(dfs, ignore_index=True)
        
        # Clean up dataframe - filter out rows with empty text
        data = data[data['clean_text'].notna() & (data['clean_text'].str.strip() != '')]
        logger.info(f"Loaded {len(data)} valid rows from {len(dfs)} files")
        
        return data
    
    def extract_document_structure(self, df: pd.DataFrame) -> Dict:
        """Extract document structure by grouping by doc_id and ordering by doc_order."""
        logger.info("Extracting document structure")
        
        # Group data by doc_id
        doc_groups = {}
        for doc_id, group in tqdm(df.groupby('doc_id'), desc="Grouping documents"):
            # Sort by doc_order to preserve the correct sequence
            ordered_group = group.sort_values('doc_order')
            doc_groups[doc_id] = ordered_group.to_dict('records')
            
            # Update stats
            chunks_count = len(ordered_group)
            self.stats["total_chunks"] += chunks_count
            self.stats["max_chunks_per_document"] = max(self.stats["max_chunks_per_document"], chunks_count)
        
        self.stats["total_documents"] = len(doc_groups)
        logger.info(f"Extracted structure for {len(doc_groups)} documents")
        
        return doc_groups
    
    def identify_parent_child_relationships(self, document_groups: Dict) -> Dict:
        """
        Identify parent-child relationships between documents
        based on doc_id patterns and prefixes.
        """
        logger.info("Identifying document relationships")
        
        # Build a relationship mapping
        relationships = {}
        doc_ids = list(document_groups.keys())
        
        # Initialize relationship structure
        for doc_id in doc_ids:
            relationships[doc_id] = {
                "parent": None,
                "children": []
            }
        
        # First approach: Check for ID prefixes
        for doc_id in tqdm(doc_ids, desc="Analyzing relationships"):
            for potential_parent in doc_ids:
                if doc_id == potential_parent:
                    continue
                    
                # If potential_parent is a prefix of doc_id
                if doc_id.startswith(potential_parent + "_"):
                    relationships[doc_id]["parent"] = potential_parent
                    relationships[potential_parent]["children"].append(doc_id)
                    break
        
        # Count documents with children
        docs_with_children = sum(1 for doc_id in relationships if relationships[doc_id]["children"])
        self.stats["documents_with_children"] = docs_with_children
        
        # Calculate max hierarchy depth
        def calculate_depth(doc_id, visited=None):
            if visited is None:
                visited = set()
            
            if doc_id in visited:  # Avoid circular references
                return 0
            
            visited.add(doc_id)
            
            if not relationships[doc_id]["children"]:
                return 1
            
            child_depths = [calculate_depth(child, visited.copy()) for child in relationships[doc_id]["children"]]
            return 1 + max(child_depths)
        
        # Calculate max depth for each root document
        root_docs = [doc_id for doc_id in relationships if not relationships[doc_id]["parent"]]
        if root_docs:
            depths = [calculate_depth(doc_id) for doc_id in root_docs]
            self.stats["max_hierarchy_depth"] = max(depths)
        
        logger.info(f"Identified {docs_with_children} documents with children")
        return relationships
    
    def generate_structured_json(self, document_groups: Dict, relationships: Dict) -> List[Dict]:
        """Generate structured records for the Hugging Face dataset."""
        logger.info("Generating structured dataset records")
        
        records = []
        
        # Process each document
        for doc_id, chunks in tqdm(document_groups.items(), desc="Creating document records"):
            # Get basic document info
            doc_title = chunks[0].get("clean_title", "") if chunks else ""
            
            # Get relationship info
            parent_id = relationships[doc_id]["parent"]
            child_ids = relationships[doc_id]["children"]
            
            # Create a structured JSON representation of the document content
            content_structured = {
                "title": doc_title,
                "sections": []
            }
            
            # Add each chunk as a section
            for i, chunk in enumerate(chunks):
                section = {
                    "order": chunk.get("doc_order", i),
                    "title": chunk.get("clean_title", ""),
                    "content": chunk.get("clean_text", ""),
                    "cid": chunk.get("cid", "")
                }
                content_structured["sections"].append(section)
            
            # Calculate hierarchy path
            hierarchy_path = self._calculate_hierarchy_path(doc_id, relationships)
            
            # Create a record for this document
            record = {
                "doc_id": doc_id,
                "title": doc_title,
                "parent_id": parent_id if parent_id else "",
                "child_ids": child_ids,
                "hierarchy_path": hierarchy_path,
                "num_chunks": len(chunks),
                "content_structured": json.dumps(content_structured),
                "text": self._generate_flattened_text(content_structured),
                "has_missing_content": any(not chunk.get("clean_text", "").strip() for chunk in chunks)
            }
            
            # Add metadata about first and last chunks for sorting
            if chunks:
                record["first_chunk_order"] = chunks[0].get("doc_order", 0)
                record["last_chunk_order"] = chunks[-1].get("doc_order", 0)
            else:
                record["first_chunk_order"] = 0
                record["last_chunk_order"] = 0
            
            records.append(record)
        
        logger.info(f"Generated {len(records)} structured document records")
        return records
    
    def _calculate_hierarchy_path(self, doc_id: str, relationships: Dict) -> List[str]:
        """Calculate the hierarchical path from root to this document."""
        path = [doc_id]
        current = doc_id
        
        # Walk up the tree to the root
        while relationships[current]["parent"]:
            parent = relationships[current]["parent"]
            path.insert(0, parent)
            current = parent
        
        return path
    
    def _generate_flattened_text(self, content_structured: Dict) -> str:
        """Generate a flattened text representation of the document."""
        lines = []
        
        # Add title
        title = content_structured.get("title", "")
        if title:
            lines.append(title)
            lines.append("")
        
        # Add sections
        for section in content_structured.get("sections", []):
            section_title = section.get("title", "")
            if section_title and section_title != title:
                lines.append(section_title)
            
            content = section.get("content", "")
            if content:
                lines.append(content)
                lines.append("")
        
        return "\n".join(lines).strip()
    
    def create_huggingface_dataset(self, records: List[Dict]) -> Dataset:
        """Create a Hugging Face dataset from records."""
        logger.info("Creating Hugging Face dataset")
        
        # Create pandas DataFrame from records
        df = pd.DataFrame(records)
        
        # Create dataset
        dataset = Dataset.from_pandas(df)
        
        logger.info(f"Created dataset with {len(dataset)} examples")
        return dataset
    
    def save_and_upload_dataset(self, dataset: Dataset, push_to_hub: bool = False):
        """Save dataset locally and optionally upload to Hugging Face Hub."""
        # Save locally
        logger.info(f"Saving dataset to {self.output_dir}")
        dataset.save_to_disk(self.output_dir)
        
        # Also save as parquet for easy loading
        parquet_path = os.path.join(self.output_dir, "structured_legal_documents.parquet")
        dataset.to_parquet(parquet_path)
        
        # Add dataset metadata
        with open(os.path.join(self.output_dir, "dataset_info.json"), "w") as f:
            json.dump({
                "description": "Structured legal documents dataset derived from the American Law dataset",
                "citation": "@dataset{structured_legal_documents, author = {tinycrops}, title = {Structured Legal Documents}}",
                "statistics": self.stats
            }, f, indent=2)
        
        logger.info(f"Dataset saved successfully with {len(dataset)} records")
        
        # Push to Hugging Face Hub if requested
        if push_to_hub:
            try:
                logger.info(f"Uploading dataset to {self.repo_id}")
                
                # Log in to Hugging Face
                token = os.environ.get("HF_TOKEN")
                if not token:
                    logger.error("HF_TOKEN environment variable not set. Cannot upload to Hub.")
                    return
                
                login(token=token)
                
                # Push dataset to Hub
                dataset.push_to_hub(
                    self.repo_id,
                    private=False,
                    token=token
                )
                
                logger.info(f"Successfully uploaded dataset to {self.repo_id}")
                
            except Exception as e:
                logger.error(f"Error uploading to Hugging Face Hub: {str(e)}")
                logger.error(traceback.format_exc())
    
    def process_and_export(self, limit_files: Optional[int] = None, push_to_hub: bool = False):
        """Process the dataset and export to Hugging Face format."""
        try:
            # Load data
            df = self.load_data(limit_files)
            
            # Extract document structure
            document_groups = self.extract_document_structure(df)
            
            # Identify relationships
            relationships = self.identify_parent_child_relationships(document_groups)
            
            # Generate structured records
            records = self.generate_structured_json(document_groups, relationships)
            
            # Create Hugging Face dataset
            dataset = self.create_huggingface_dataset(records)
            
            # Save and upload
            self.save_and_upload_dataset(dataset, push_to_hub=push_to_hub)
            
            logger.info("Dataset export completed successfully")
            logger.info(f"Dataset statistics: {self.stats}")
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            logger.error(traceback.format_exc())

def main():
    """Main function to execute the dataset export process."""
    logger.info("Starting legal dataset export")
    
    # Set parameters
    input_dir = "processed_data"
    output_dir = "structured_legal_dataset"
    repo_id = "tinycrops/structured_legal_documents"
    
    # Optional: Set limit_files to a number to process only a subset of files
    limit_files = None  # Process all files
    
    # Whether to push to Hugging Face Hub
    push_to_hub = False  # Change to True to push to Hub
    
    # Create exporter and process dataset
    exporter = LegalDatasetExporter(input_dir, output_dir, repo_id)
    exporter.process_and_export(limit_files=limit_files, push_to_hub=push_to_hub)
    
    logger.info("Dataset export process completed")

if __name__ == "__main__":
    main()