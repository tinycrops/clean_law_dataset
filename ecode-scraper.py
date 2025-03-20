import csv
import os
import time
import sys
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException

def setup_driver(headless=False):
    """
    Set up and return a configured Chrome WebDriver.
    
    Args:
        headless: Whether to run Chrome in headless mode
    """
    chrome_options = Options()
    
    if headless:
        chrome_options.add_argument("--headless")
    
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")  # Hide automation
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-notifications")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)
    chrome_options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36")
    
    driver = webdriver.Chrome(options=chrome_options)
    
    # Set navigator.webdriver to undefined using CDP
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    driver.set_page_load_timeout(60)
    return driver

def read_urls_from_csv(filename: str, column_name: str, start_index: int = 0, limit: int = None) -> List[Dict[str, str]]:
    """
    Read URLs and related data from a CSV file.
    
    Args:
        filename: Path to the CSV file
        column_name: Name of the column containing URLs
        start_index: Index to start reading from (to resume scraping)
        limit: Maximum number of URLs to read (None for all)
        
    Returns:
        List of dictionaries with row data
    """
    rows = []
    try:
        with open(filename, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i < start_index:
                    continue
                if limit is not None and len(rows) >= limit:
                    break
                if column_name in row and row[column_name]:
                    rows.append(row)
    except Exception as e:
        print(f"Error reading CSV file: {str(e)}")
    return rows

def save_content_to_file(code_id: str, content: str) -> str:
    """
    Save the HTML content to a file.
    
    Args:
        code_id: Identifier for the code
        content: HTML content to save
        
    Returns:
        Path to the saved file
    """
    os.makedirs("downloaded_codes", exist_ok=True)
    file_path = f"downloaded_codes/{code_id}.html"
    
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)
    
    return file_path

def save_results_to_csv(results: List[Dict[str, Any]], output_filename: str = "ecode_results.csv") -> None:
    """
    Save results to a CSV file.
    
    Args:
        results: List of result dictionaries
        output_filename: Output CSV filename
    """
    if not results:
        print("No results to save")
        return
    
    # Check if file exists to append rather than overwrite
    file_exists = os.path.isfile(output_filename)
    
    with open(output_filename, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        if not file_exists:
            writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_filename}")

def save_state(current_index, cookies=None):
    """Save the current state to a file to allow resuming later"""
    state = {
        "current_index": current_index,
        "timestamp": datetime.now().isoformat(),
        "cookies": cookies
    }
    
    with open("scraper_state.json", "w") as f:
        json.dump(state, f)
    
    print(f"State saved: index {current_index}")

def load_state():
    """Load the previous state if available"""
    try:
        if os.path.exists("scraper_state.json"):
            with open("scraper_state.json", "r") as f:
                state = json.loads(f.read())
                print(f"Loaded state: index {state.get('current_index', 0)}")
                return state
    except Exception as e:
        print(f"Error loading state: {e}")
    
    return {"current_index": 0, "cookies": None}

def interactive_scrape(start_index=0, limit=None):
    """
    Interactive scraping function that allows manual intervention for CAPTCHA.
    """
    # Read URLs from CSV
    print("Reading URLs from CSV...")
    rows = read_urls_from_csv("ecode_urls.csv", "print_page", start_index, limit)
    
    if not rows:
        print("No URLs found in the CSV file")
        return
    
    print(f"Found {len(rows)} URLs to process")
    
    # Initialize WebDriver in non-headless mode
    print("Setting up the WebDriver in interactive mode...")
    driver = setup_driver(headless=False)
    
    results = []
    
    try:
        # Initialize with the main site
        print("Initializing session with eCode360...")
        try:
            driver.get("https://ecode360.com/")
            input("Press Enter after the main page loads and you've accepted any cookies or dismissed any popups...")
            
            # Store cookies after initialization
            cookies = driver.get_cookies()
            save_state(start_index, cookies)
            
        except Exception as e:
            print(f"Warning: Could not initialize session: {str(e)}")
        
        # Process each URL with manual intervention
        for i, row in enumerate(rows):
            url = row["print_page"]
            current_index = start_index + i
            
            print(f"\nProcessing URL {current_index+1}: {url}")
            print(f"Municipality: {row.get('place_name', 'Unknown')}")
            
            try:
                # Navigate to the URL
                driver.get(url)
                
                # Check for CAPTCHA or verification
                captcha_prompt = input("Do you need to solve a CAPTCHA? (y/n): ")
                
                if captcha_prompt.lower() == 'y':
                    input("Solve the CAPTCHA, then press Enter when done...")
                
                # Additional wait for user to verify content has loaded completely
                input(f"Press Enter after content has loaded for {url}...")
                
                # Get the page source after user interaction
                html_content = driver.page_source
                
                # Extract code identifier from URL
                code_id = url.split('/')[-1].split('?')[0] if '?' in url else url.split('/')[-1]
                
                # Save the content
                file_path = save_content_to_file(code_id, html_content)
                print(f"Saved content to {file_path}")
                
                result = {
                    "url": url,
                    "code_id": code_id,
                    "municipality": row.get("place_name", ""),
                    "gnis": row.get("gnis", ""),
                    "status_code": 200,  # Assuming success
                    "content_length": len(html_content),
                    "file_path": file_path,
                    "success": True,
                    "timestamp": datetime.now().isoformat()
                }
                
            except Exception as e:
                print(f"Error processing {url}: {str(e)}")
                result = {
                    "url": url,
                    "code_id": url.split('/')[-1].split('?')[0] if '?' in url else url.split('/')[-1],
                    "municipality": row.get("place_name", ""),
                    "gnis": row.get("gnis", ""),
                    "status_code": 500,
                    "content_length": 0,
                    "success": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
            
            results.append(result)
            
            # Save results incrementally
            save_results_to_csv([result])
            
            # Save state for resume capability
            save_state(current_index + 1)
            
            # Ask if user wants to continue
            if i < len(rows) - 1:
                cont = input("Continue to next URL? (y/n): ")
                if cont.lower() != 'y':
                    print("Stopping at user request.")
                    break
        
        # Print summary
        successful = sum(1 for r in results if r["success"])
        print(f"Successfully fetched {successful} out of {len(rows)} URLs")
        print(f"Downloaded content saved to the 'downloaded_codes' directory")
        
    finally:
        # Ask before closing the browser
        input("Press Enter to close the browser...")
        driver.quit()
        print("WebDriver closed")

def main():
    print("eCode360 Interactive Scraper with CAPTCHA Handler")
    print("=" * 50)
    print("This script will open a Chrome browser window.")
    print("You will need to manually solve CAPTCHAs when they appear.")
    print("The script will save your progress so you can resume later.")
    print("=" * 50)
    
    # Load previous state if available
    state = load_state()
    start_index = state.get("current_index", 0)
    
    if start_index > 0:
        resume = input(f"Resume from URL #{start_index+1}? (y/n): ")
        if resume.lower() != 'y':
            start_index = 0
    
    try:
        limit_input = input("How many URLs would you like to process? (press Enter for all): ")
        limit = int(limit_input) if limit_input.strip() else None
    except ValueError:
        print("Invalid input. Using all URLs.")
        limit = None
    
    interactive_scrape(start_index, limit)

if __name__ == "__main__":
    main()