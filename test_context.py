import pandas as pd
import re
from create_citation_dataset import extract_relevant_text

def test_context_extraction():
    """Test the context extraction function with some sample cases"""
    
    # Create a sample citation
    citation_match = {
        'full_citation': '17 U.S.C. 501',
        'citation_type': 'USC',
        'title': '17',
        'section': '501'
    }
    
    # Case 1: Citation at the start of text
    html_1 = "17 U.S.C. 501 prohibits copyright infringement. This is further explained in the Copyright Act."
    context_1 = extract_relevant_text(html_1, citation_match)
    print("\nCase 1 - Citation at start:")
    print(context_1)
    
    # Case 2: Citation in the middle of text
    html_2 = "Copyright infringement is specifically prohibited by 17 U.S.C. 501 as outlined in the Copyright Act."
    context_2 = extract_relevant_text(html_2, citation_match)
    print("\nCase 2 - Citation in middle:")
    print(context_2)
    
    # Case 3: Citation at the end of text
    html_3 = "For claims of copyright infringement, plaintiffs must refer to 17 U.S.C. 501"
    context_3 = extract_relevant_text(html_3, citation_match)
    print("\nCase 3 - Citation at end:")
    print(context_3)
    
    # Case 4: Citation with preceding sentence
    html_4 = "The Copyright Act provides legal protection for original works. Infringement is prohibited under 17 U.S.C. 501 as stated in the law."
    context_4 = extract_relevant_text(html_4, citation_match)
    print("\nCase 4 - Citation with preceding sentence:")
    print(context_4)
    
    # Check if any contexts appear cut off at the beginning
    for i, context in enumerate([context_1, context_2, context_3, context_4], 1):
        if re.match(r'^[a-z]', context) and not re.match(r'^[a-z]+\s+U\.S\.C', context):
            print(f"\nWarning: Context {i} may be cut off at the beginning: {context[:50]}...")
        else:
            print(f"\nContext {i} appears intact at the beginning")

if __name__ == "__main__":
    test_context_extraction() 