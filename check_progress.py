import os
import pandas as pd
import argparse
import time
from datetime import datetime

def check_progress(dataset_dir="american_law_full"):
    """Check the progress of the dataset processing
    
    Args:
        dataset_dir: Directory containing the dataset
    """
    csv_path = os.path.join(dataset_dir, "citations.csv")
    
    if not os.path.exists(csv_path):
        print(f"No progress file found at {csv_path}")
        return
    
    try:
        df = pd.read_csv(csv_path)
        total_citations = len(df)
        
        # Get citation types
        type_counts = df['citation_type'].value_counts().to_dict()
        
        print(f"Progress check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total citations processed: {total_citations}")
        print("Citation counts by type:")
        for citation_type, count in type_counts.items():
            print(f"  {citation_type}: {count}")
        
        # Sample of last 5 citations
        print("\nLast 5 citations:")
        for i in range(min(5, len(df))):
            idx = len(df) - i - 1
            row = df.iloc[idx]
            print(f"  {row['citation_type']}: {row['full_citation']}")
            print(f"  Context start: {row['context'][:100].replace(chr(10), ' ')}")
            print()
        
    except Exception as e:
        print(f"Error checking progress: {e}")

def monitor_progress(dataset_dir="american_law_full", interval=300):
    """Continuously monitor the progress
    
    Args:
        dataset_dir: Directory containing the dataset
        interval: Check interval in seconds
    """
    try:
        while True:
            check_progress(dataset_dir)
            print(f"Next check in {interval} seconds. Press Ctrl+C to stop monitoring.")
            print("-" * 80)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Monitoring stopped.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check progress of dataset processing")
    parser.add_argument("--dir", type=str, default="american_law_full", help="Directory containing the dataset")
    parser.add_argument("--monitor", action="store_true", help="Continuously monitor progress")
    parser.add_argument("--interval", type=int, default=300, help="Monitoring interval in seconds")
    
    args = parser.parse_args()
    
    if args.monitor:
        monitor_progress(args.dir, args.interval)
    else:
        check_progress(args.dir) 