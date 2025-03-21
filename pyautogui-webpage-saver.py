import pyautogui
import pandas as pd
import time
import webbrowser
import os
import sys
from tqdm import tqdm

def save_webpages_from_csv(csv_path, column_name='full_page', delay_after_load=5, delay_after_save=2):
    """
    Script to automatically visit URLs from a CSV file and save each page using keyboard shortcuts.
    
    Args:
        csv_path (str): Path to the CSV file containing URLs
        column_name (str): Name of the column containing the URLs to visit
        delay_after_load (int): Seconds to wait after loading the page before saving
        delay_after_save (int): Seconds to wait after saving before moving to the next URL
    """
    try:
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
        print(f"Found {len(urls)} URLs to process.")
        
        # Ask for confirmation
        print("\nThis script will:")
        print("1. Open each URL in your default browser")
        print("2. Wait for the page to load")
        print("3. Press Ctrl+S to save the page")
        print("4. Close the browser and move to the next URL")
        
        confirmation = input("\nReady to proceed? (yes/no): ")
        if confirmation.lower() not in ['yes', 'y']:
            print("Operation cancelled.")
            return
            
        # Process each URL
        for i, url in enumerate(tqdm(urls, desc="Processing URLs")):
            print(f"\n[{i+1}/{len(urls)}] Processing: {url}")
            
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
            
            # Press Enter to save with the default name
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
            
            # # Close the browser tab (Ctrl+W)
            # print("Closing the browser tab...")
            # pyautogui.hotkey('ctrl', 'w')
            
            # Add a small delay before the next URL
            time.sleep(1)
            
        print("\nAll URLs have been processed successfully!")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = input("Enter the path to your CSV file: ")
    
    # Allow customization of delays
    try:
        load_delay = int(input("Enter delay after page load (seconds, default 5): ") or 5)
        save_delay = int(input("Enter delay after saving (seconds, default 2): ") or 2)
    except ValueError:
        print("Invalid input for delay. Using default values.")
        load_delay = 5
        save_delay = 2
    
    column_name = input("Enter the column name containing URLs (default 'full_page'): ") or 'full_page'
    
    save_webpages_from_csv(csv_path, column_name, load_delay, save_delay)