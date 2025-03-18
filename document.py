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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("legal_document_generator.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("legal_document_generator")

class LegalDocumentGenerator:
    """
    Generate structured legal documents from the American Law dataset
    by using doc_id and doc_order to reconstruct the hierarchical structure.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """Initialize the generator with input and output directories."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Track statistics
        self.stats = {
            "total_rows": 0,
            "documents_processed": 0,
            "hierarchical_documents": 0,
            "standalone_documents": 0
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
        self.stats["total_rows"] = len(data)
        
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
        
        logger.info(f"Extracted structure for {len(doc_groups)} documents")
        self.stats["documents_processed"] = len(doc_groups)
        
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
        
        # Count hierarchical and standalone documents
        hierarchical_docs = set()
        for doc_id, rel in relationships.items():
            if rel["parent"] or rel["children"]:
                hierarchical_docs.add(doc_id)
        
        self.stats["hierarchical_documents"] = len(hierarchical_docs)
        self.stats["standalone_documents"] = len(document_groups) - len(hierarchical_docs)
        
        logger.info(f"Identified {len(hierarchical_docs)} documents in hierarchical relationships")
        return relationships
    
    def generate_document_tree(self, document_groups: Dict, relationships: Dict) -> Dict:
        """Generate a nested document tree based on the identified relationships."""
        logger.info("Generating document tree")
        
        # Find root documents (those with no parent)
        root_docs = {doc_id for doc_id in document_groups if not relationships[doc_id]["parent"]}
        logger.info(f"Found {len(root_docs)} root documents")
        
        # Build tree recursively
        def build_subtree(doc_id):
            doc_data = document_groups[doc_id]
            children = relationships[doc_id]["children"]
            
            # Create document node
            node = {
                "id": doc_id,
                "title": doc_data[0]["clean_title"] if doc_data else "Untitled Document",
                "content_chunks": doc_data,
                "children": [build_subtree(child_id) for child_id in children]
            }
            return node
        
        # Build the complete tree
        tree = {root_id: build_subtree(root_id) for root_id in root_docs}
        
        return tree
    
    def export_documents(self, document_tree: Dict):
        """Export documents to the output directory in a structured format."""
        logger.info(f"Exporting documents to {self.output_dir}")
        
        # Create output structure
        structured_dir = os.path.join(self.output_dir, "structured")
        flat_dir = os.path.join(self.output_dir, "flat")
        os.makedirs(structured_dir, exist_ok=True)
        os.makedirs(flat_dir, exist_ok=True)
        
        # Generate index data
        index = {
            "stats": self.stats,
            "documents": {}
        }
        
        # Process each root document
        for root_id, root_node in tqdm(document_tree.items(), desc="Exporting documents"):
            # Export document and its children recursively
            self._export_document_recursive(root_node, structured_dir, "", index)
            
            # Also export as flat document for convenience
            self._export_flat_document(root_node, flat_dir, index)
        
        # Save index file
        with open(os.path.join(self.output_dir, "document_index.json"), "w", encoding="utf-8") as f:
            json.dump(index, f, indent=4)
        
        logger.info(f"Successfully exported {len(document_tree)} root documents with their hierarchies")
        logger.info(f"Export statistics: {self.stats}")
    
    def _export_document_recursive(self, node: Dict, base_dir: str, path_prefix: str, index: Dict):
        """Recursively export a document and its children."""
        doc_id = node["id"]
        doc_dir = os.path.join(base_dir, path_prefix, doc_id)
        os.makedirs(doc_dir, exist_ok=True)
        
        # Generate document content
        content = self._format_document_content(node)
        
        # Save document
        doc_path = os.path.join(doc_dir, f"{doc_id}.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)
        
        # Save metadata
        metadata = {
            "id": doc_id,
            "title": node["title"],
            "chunks_count": len(node["content_chunks"]),
            "children_count": len(node["children"]),
            "path": os.path.join(path_prefix, doc_id, f"{doc_id}.md") if path_prefix else os.path.join(doc_id, f"{doc_id}.md")
        }
        
        with open(os.path.join(doc_dir, "metadata.json"), "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=4)
        
        # Add to index
        index["documents"][doc_id] = metadata
        
        # Process children
        for child in node["children"]:
            new_path = os.path.join(path_prefix, doc_id) if path_prefix else doc_id
            self._export_document_recursive(child, base_dir, new_path, index)
    
    def _export_flat_document(self, node: Dict, flat_dir: str, index: Dict):
        """Export a document in flat format (including all child content)."""
        doc_id = node["id"]
        
        # Generate combined content for this node and all descendants
        content = self._format_document_content(node, include_children=True)
        
        # Save document
        doc_path = os.path.join(flat_dir, f"{doc_id}.md")
        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(content)
    
    def _format_document_content(self, node: Dict, include_children: bool = False) -> str:
        """Format document content as Markdown."""
        doc_id = node["id"]
        title = node["title"]
        chunks = node["content_chunks"]
        
        # Start with document title
        lines = [f"# {title}", ""]
        
        # Add metadata
        lines.extend([
            "## Document Metadata",
            f"- **Document ID:** {doc_id}",
            f"- **Number of Content Chunks:** {len(chunks)}",
            f"- **Number of Child Documents:** {len(node['children'])}",
            ""
        ])
        
        # Add content chunks
        lines.append("## Content")
        
        for i, chunk in enumerate(chunks):
            # Add section title if different from document title and not empty
            chunk_title = chunk.get("clean_title", "").strip()
            if chunk_title and chunk_title != title:
                lines.append(f"### {chunk_title}")
            
            # Add content
            content = chunk.get("clean_text", "").strip()
            if content:
                lines.append(content)
                lines.append("")  # Empty line after content
        
        # Include child documents if requested
        if include_children and node["children"]:
            lines.append("## Child Documents")
            
            for child in node["children"]:
                # Add child document as a subsection
                child_content = self._format_document_content(child, include_children=True)
                # Increase heading levels by 1
                child_content = re.sub(r'^(#+)', r'#\1', child_content, flags=re.MULTILINE)
                lines.append(child_content)
        
        return "\n".join(lines)
    
    def process_dataset(self, limit_files: Optional[int] = None, sample_docs: Optional[int] = None):
        """Process the dataset to generate structured legal documents."""
        try:
            # Load data
            df = self.load_data(limit_files)
            
            # Optionally sample a subset of documents
            if sample_docs:
                doc_ids = df['doc_id'].unique()
                if len(doc_ids) > sample_docs:
                    sampled_ids = np.random.choice(doc_ids, sample_docs, replace=False)
                    df = df[df['doc_id'].isin(sampled_ids)]
                    logger.info(f"Sampled {len(sampled_ids)} document IDs for processing")
            
            # Extract document structure
            document_groups = self.extract_document_structure(df)
            
            # Identify relationships
            relationships = self.identify_parent_child_relationships(document_groups)
            
            # Generate document tree
            document_tree = self.generate_document_tree(document_groups, relationships)
            
            # Export documents
            self.export_documents(document_tree)
            
            logger.info("Document generation completed successfully")
            
        except Exception as e:
            logger.error(f"Error processing dataset: {str(e)}")
            logger.error(traceback.format_exc())

def main():
    """Main function to execute the document generation process."""
    logger.info("Starting legal document generation")
    
    # Set parameters
    input_dir = "processed_data"
    output_dir = "generated_documents"
    
    # Sample a subset of files/documents for faster processing
    # Set to None to process the entire dataset
    limit_files = None  # Process only the first 10 parquet files
    sample_docs = None  # Sample 200 documents
    
    # Create generator and process dataset
    generator = LegalDocumentGenerator(input_dir, output_dir)
    generator.process_dataset(limit_files=limit_files, sample_docs=sample_docs)
    
    logger.info("Document generation process completed")

if __name__ == "__main__":
    main()