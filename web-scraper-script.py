import os
import pandas as pd
import time
import webbrowser
import pyautogui
import glob
import sys
import subprocess
from tqdm import tqdm
import argparse

def save_webpages_from_csv(csv_path, column_name='full_page', delay_after_load=5, delay_after_save=2,
                          output_dir='saved_pages', start_index=0, end_index=None):
    """
    Script to automatically visit URLs from a CSV file and save each page using keyboard shortcuts.
    
    Args:
        csv_path (str): Path to the CSV file containing URLs
        column_name (str): Name of the column containing the URLs to visit
        delay_after_load (int): Seconds to wait after loading the page before saving
        delay_after_save (int): Seconds to wait after saving before moving to the next URL
        output_dir (str): Directory to save the downloaded pages
        start_index (int): Index to start processing from (useful for resuming)
        end_index (int): Index to stop processing at (None means process all)
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Load the CSV file
        print(f"Loading URLs from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Check if the column exists
        if column_name not in df.columns:
            print(f"Error: Column '{column_name}' not found in the CSV file.")
            print(f"Available columns: {', '.join(df.columns)}")
            return
        
        # Get the list of URLs
        urls = df[column_name].dropna().tolist()
        
        # Apply start and end indices
        if end_index is None:
            end_index = len(urls)
        urls_to_process = urls[start_index:end_index]
        
        print(f"Found {len(urls)} URLs, processing {len(urls_to_process)} (from index {start_index} to {end_index-1})")
        
        # Ask for confirmation
        print("\nThis script will:")
        print("1. Open each URL in your default browser")
        print("2. Wait for the page to load")
        print("3. Press Ctrl+S to save the page")
        print("4. Save the page to the specified output directory")
        print("5. Close the browser and move to the next URL")
        
        confirmation = input("\nReady to proceed? (yes/no): ")
        if confirmation.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
            
        # Process each URL
        for i, url in enumerate(tqdm(urls_to_process, desc="Processing URLs")):
            global_index = i + start_index
            print(f"\n[{global_index+1}/{len(urls)}] Processing: {url}")
            
            # Extract city name from URL for use in the filename
            # Example URL: https://ecode360.com/print/LO4963?guid=LO4963
            city_code = url.split('/print/')[1].split('?')[0] if '/print/' in url else 'unknown'
            
            # Check if this URL has already been processed
            existing_files = glob.glob(f"{output_dir}/{city_code}*.html")
            if existing_files:
                print(f"Skipping {url} - already saved as {existing_files[0]}")
                continue
            
            # Open the URL in the default browser
            webbrowser.open(url)
            
            # Wait for the page to load
            print(f"Waiting {delay_after_load} seconds for page to load...")
            time.sleep(delay_after_load)
            
            # Press Ctrl+S to save
            print("Pressing Ctrl+S to save the page...")
            pyautogui.hotkey('ctrl', 's')
            
            # Wait for the save dialog to appear
            time.sleep(2)
            
            # Type the output path and filename
            filename = f"{city_code} - {global_index+1}.html"
            save_path = os.path.join(output_dir, filename)
            print(f"Saving as: {save_path}")
            
            # Type the full path
            pyautogui.typewrite(save_path)
            time.sleep(1)
            
            # Press Enter to save
            print("Pressing Enter to confirm save...")
            pyautogui.press('enter')
            
            # Wait a moment and press Enter again to confirm any additional dialogs
            time.sleep(1)
            print("Pressing Enter again to confirm any additional dialogs...")
            pyautogui.press('enter')
            
            # Press Esc to dismiss any remaining dialogs
            time.sleep(1)
            print("Pressing Esc to dismiss any remaining dialogs...")
            pyautogui.press('esc')
            
            # Wait after saving
            print(f"Waiting {delay_after_save} seconds after saving...")
            time.sleep(delay_after_save)
            
            # Close the browser tab (Ctrl+W)
            print("Closing the browser tab...")
            pyautogui.hotkey('ctrl', 'w')
            
            # Add a small delay before the next URL
            time.sleep(1)
            
        print("\nAll URLs have been processed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def process_saved_pages(input_dir, output_file):
    """
    Process all saved HTML pages in the specified directory and extract charter information
    
    Args:
        input_dir (str): Directory containing saved HTML files
        output_file (str): Path to save the output parquet file
    """
    # Check if extractor script exists
    extractor_script = "final-charter-extractor.py"
    if not os.path.exists(extractor_script):
        print(f"Error: Extractor script {extractor_script} not found.")
        return False
    
    # Run the extractor on the directory
    print(f"Processing HTML files in {input_dir}...")
    try:
        subprocess.run([sys.executable, extractor_script, input_dir, output_file], check=True)
        print(f"Processing completed. Results saved to {output_file}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running extractor: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Web scraping and processing pipeline for charter documents")
    
    # Main operation mode
    parser.add_argument("--mode", choices=["scrape", "process", "full"], default="full",
                       help="Operation mode: scrape pages, process saved pages, or full pipeline")
    
    # Scraping parameters
    parser.add_argument("--csv", help="Path to CSV file containing URLs")
    parser.add_argument("--column", default="full_page", help="Column name containing URLs")
    parser.add_argument("--load-delay", type=int, default=5, help="Delay after page load (seconds)")
    parser.add_argument("--save-delay", type=int, default=2, help="Delay after saving (seconds)")
    parser.add_argument("--output-dir", default="saved_pages", help="Directory to save HTML pages")
    parser.add_argument("--start", type=int, default=0, help="Start index for URL processing")
    parser.add_argument("--end", type=int, default=None, help="End index for URL processing")
    
    # Processing parameters
    parser.add_argument("--input-dir", help="Directory containing saved HTML files (defaults to output-dir)")
    parser.add_argument("--output", default="charter_data.parquet", help="Output parquet file path")
    
    args = parser.parse_args()
    
    # Set input directory to output directory if not specified
    if args.input_dir is None:
        args.input_dir = args.output_dir
    
    # Execute based on mode
    if args.mode in ["scrape", "full"]:
        if args.csv is None:
            parser.error("--csv is required for scrape or full mode")
        
        save_webpages_from_csv(
            csv_path=args.csv,
            column_name=args.column,
            delay_after_load=args.load_delay,
            delay_after_save=args.save_delay,
            output_dir=args.output_dir,
            start_index=args.start,
            end_index=args.end
        )
    
    if args.mode in ["process", "full"]:
        process_saved_pages(args.input_dir, args.output)

if __name__ == "__main__":
    main()
