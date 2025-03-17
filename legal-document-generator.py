import pandas as pd
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from bs4 import BeautifulSoup
import re
import logging
from tqdm import tqdm

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
    A class for generating structured legal documents from the American Law dataset.
    This reconstructs the hierarchical structure based on doc_id and doc_order.
    """
    
    def __init__(self, input_dir: str, output_dir: str):
        """
        Initialize the LegalDocumentGenerator.
        
        Args:
            input_dir: Directory containing processed parquet files
            output_dir: Directory to save generated documents
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.document_tree = {}  # doc_id -> document tree
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text by removing HTML tags and normalizing whitespace.
        
        Args:
            text: HTML text to clean
            
        Returns:
            Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Parse HTML content
        soup = BeautifulSoup(text, 'lxml')
        
        # Extract text without HTML tags
        clean_text = soup.get_text()
        
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text
    
    def _extract_document_id_components(self, doc_id: str) -> Tuple[str, List[str]]:
        """
        Extract components from a document ID.
        
        Args:
            doc_id: Document ID string
            
        Returns:
            Tuple containing (base_id, path_components)
        """
        # Try to identify common patterns in doc_id to split into components
        # This is a simplified approach and might need refinement based on actual data
        
        # First, check if it contains underscores which often separate components
        if '_' in doc_id:
            parts = doc_id.split('_')
            base_id = parts[0]
            components = parts[1:]
            return base_id, components
        
        # If no clear separator, try to split based on common patterns
        # For example, if doc_id is something like "COOR_CH1GEPR_S1-14SUCO"
        # we might split at transitions between letter types
        
        # Default fallback
        return doc_id, []
    
    def _create_document_hierarchy(self, df: pd.DataFrame) -> Dict:
        """
        Create a document hierarchy based on doc_id and doc_order.
        
        Args:
            df: DataFrame containing legal document data
            
        Returns:
            Dictionary representing the document hierarchy
        """
        hierarchy = {}
        
        # Sort by doc_id and doc_order to ensure correct ordering
        df = df.sort_values(by=['doc_id', 'doc_order'])
        
        # Group by doc_id
        for doc_id, group in df.groupby('doc_id'):
            # Create a document node
            doc_node = {
                'id': doc_id,
                'title': None,  # Will set from first chunk
                'sections': [],
                'metadata': {
                    'num_chunks': len(group)
                }
            }
            
            # Sort chunks by doc_order
            chunks = group.sort_values('doc_order')
            
            # Process each chunk
            for _, chunk in chunks.iterrows():
                section = {
                    'order': chunk['doc_order'],
                    'title': chunk['clean_title'],
                    'content': chunk['clean_text'],
                    'cid': chunk['cid']
                }
                doc_node['sections'].append(section)
            
            # Set document title from first chunk if available
            if doc_node['sections']:
                doc_node['title'] = doc_node['sections'][0]['title']
            
            # Add to hierarchy
            hierarchy[doc_id] = doc_node
        
        return hierarchy
    
    def _analyze_document_relationships(self, hierarchy: Dict) -> Dict:
        """
        Analyze relationships between documents to create a nested structure.
        
        Args:
            hierarchy: Flat document hierarchy
            
        Returns:
            Dictionary with document relationships
        """
        relationships = {}
        
        # Extract document components for analysis
        doc_components = {}
        for doc_id in hierarchy:
            base_id, components = self._extract_document_id_components(doc_id)
            doc_components[doc_id] = {
                'base_id': base_id,
                'components': components
            }
        
        # Identify potential parent-child relationships
        for doc_id, components_info in doc_components.items():
            base_id = components_info['base_id']
            components = components_info['components']
            
            # Initialize relationships entry
            if doc_id not in relationships:
                relationships[doc_id] = {
                    'parent': None,
                    'children': []
                }
            
            # Look for parent documents
            for other_id, other_info in doc_components.items():
                if doc_id == other_id:
                    continue
                
                other_base = other_info['base_id']
                other_components = other_info['components']
                
                # Check if one could be the parent of the other
                if base_id == other_base and len(components) > len(other_components):
                    # Check if components are a superset
                    if all(c in components for c in other_components):
                        # other_id could be a parent of doc_id
                        relationships[doc_id]['parent'] = other_id
                        
                        # Ensure the other document has this as a child
                        if other_id not in relationships:
                            relationships[other_id] = {
                                'parent': None,
                                'children': []
                            }
                        relationships[other_id]['children'].append(doc_id)
        
        return relationships
    
    def _generate_markdown_document(self, doc_node: Dict) -> str:
        """
        Generate a markdown document from a document node.
        
        Args:
            doc_node: Document node
            
        Returns:
            Markdown content as a string
        """
        markdown = []
        
        # Add document title
        title = doc_node.get('title', 'Untitled Document')
        markdown.append(f"# {title}\n")
        
        # Add document metadata
        markdown.append("## Document Metadata\n")
        markdown.append(f"- Document ID: {doc_node['id']}\n")
        markdown.append(f"- Number of Sections: {len(doc_node['sections'])}\n\n")
        
        # Add sections
        markdown.append("## Content\n")
        
        for section in doc_node['sections']:
            section_title = section.get('title', '')
            if section_title:
                markdown.append(f"### {section_title}\n")
            
            content = section.get('content', '')
            if content:
                markdown.append(f"{content}\n\n")
            else:
                markdown.append("*No content available*\n\n")
        
        return '\n'.join(markdown)
    
    def load_data(self) -> pd.DataFrame:
        """
        Load data from parquet files.
        
        Returns:
            DataFrame containing all loaded data
        """
        logger.info(f"Loading data from {self.input_dir}")
        
        # Get all parquet files
        parquet_files = list(Path(self.input_dir).glob("*.parquet"))
        
        if not parquet_files:
            logger.error(f"No parquet files found in {self.input_dir}")
            raise FileNotFoundError(f"No parquet files found in {self.input_dir}")
        
        logger.info(f"Found {len(parquet_files)} parquet files")
        
        # Load data from parquet files
        dfs = []
        for file in tqdm(parquet_files, desc="Loading files"):
            try:
                df = pd.read_parquet(file)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")
        
        if not dfs:
            logger.error("No data loaded from parquet files")
            raise ValueError("No data loaded from parquet files")
        
        # Combine all dataframes
        combined_df = pd.concat(dfs, ignore_index=True)
        
        logger.info(f"Loaded {len(combined_df)} rows from {len(dfs)} files")
        
        return combined_df
    
    def generate_documents(self, sample_size: Optional[int] = None):
        """
        Generate documents from the data.
        
        Args:
            sample_size: Optional number of documents to sample for processing
        """
        # Load data
        df = self.load_data()
        
        # Remove rows with blank text
        non_blank_df = df[df['clean_text'].notna() & (df['clean_text'] != '')]
        logger.info(f"Removed {len(df) - len(non_blank_df)} blank rows, {len(non_blank_df)} rows remaining")
        df = non_blank_df
        
        # Get unique document IDs
        doc_ids = df['doc_id'].unique()
        logger.info(f"Found {len(doc_ids)} unique document IDs")
        
        # Optionally sample a subset of documents
        if sample_size and sample_size < len(doc_ids):
            logger.info(f"Sampling {sample_size} documents")
            doc_ids = np.random.choice(doc_ids, size=sample_size, replace=False)
        
        # Create document hierarchy
        logger.info("Creating document hierarchy")
        hierarchy = self._create_document_hierarchy(df[df['doc_id'].isin(doc_ids)])
        
        # Analyze document relationships
        logger.info("Analyzing document relationships")
        relationships = self._analyze_document_relationships(hierarchy)
        
        # Generate documents
        logger.info("Generating documents")
        
        # Create a directory for each document
        for doc_id, doc_node in tqdm(hierarchy.items(), desc="Generating documents"):
            # Create document directory
            doc_dir = os.path.join(self.output_dir, doc_id)
            os.makedirs(doc_dir, exist_ok=True)
            
            # Generate markdown document
            markdown = self._generate_markdown_document(doc_node)
            
            # Save markdown document
            with open(os.path.join(doc_dir, f"{doc_id}.md"), 'w', encoding='utf-8') as f:
                f.write(markdown)
            
            # Save document metadata with relationship information
            metadata = {
                'id': doc_id,
                'title': doc_node['title'],
                'num_sections': len(doc_node['sections']),
                'relationships': relationships.get(doc_id, {'parent': None, 'children': []})
            }
            
            with open(os.path.join(doc_dir, "metadata.json"), 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
        
        # Generate an index of all documents
        logger.info("Generating document index")
        index = {
            'documents': [],
            'total_documents': len(hierarchy),
            'relationships': relationships
        }
        
        for doc_id, doc_node in hierarchy.items():
            index['documents'].append({
                'id': doc_id,
                'title': doc_node['title'],
                'path': f"{doc_id}/{doc_id}.md",
                'num_sections': len(doc_node['sections'])
            })
        
        # Save index
        with open(os.path.join(self.output_dir, "document_index.json"), 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        
        logger.info(f"Generated {len(hierarchy)} documents in {self.output_dir}")

def main():
    """
    Main function to execute the document generation process.
    """
    logger.info("Starting legal document generation")
    
    # Set paths
    input_dir = "processed_data"
    output_dir = "generated_documents"
    
    # Create generator
    generator = LegalDocumentGenerator(input_dir, output_dir)
    
    # Generate documents
    try:
        # Use a small sample size for testing
        # Set to None to process all documents
        generator.generate_documents(sample_size=None)
    except Exception as e:
        logger.error(f"Error generating documents: {e}", exc_info=True)
    
    logger.info("Document generation complete")

if __name__ == "__main__":
    main()
