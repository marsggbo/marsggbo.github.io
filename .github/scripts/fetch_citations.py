import json
import os
from datetime import datetime
from scholarly import scholarly

AUTHOR_ID = os.environ.get('SCHOLAR_USER_ID', 'LYNKm_8AAAAJ')
JSON_PATH = 'citations.json'

def main():
    print(f"Fetching data for author: {AUTHOR_ID}")
    try:
        author = scholarly.search_author_id(AUTHOR_ID)
        scholarly.fill(author, sections=['publications'])
    except Exception as e:
        print(f"Error fetching data from Google Scholar: {e}")
        return

    today = datetime.now().strftime('%Y-%m-%d')
    
    # Load existing JSON or initialize
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Make sure it's a dict, not the old list-based format
            if 'dates' in data and 'papers' in data:
                data = {} # Clear old format
    else:
        data = {}
    
    # Sync new data
    for pub in author['publications']:
        title = pub['bib'].get('title', 'Unknown Title')
        citations = pub.get('num_citations', 0)
        
        if title not in data:
            data[title] = {
                "citations": {}
            }
        
        # Update today's entry
        data[title]["citations"][today] = citations

    # Save format
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {JSON_PATH} with {len(data)} papers (Date: {today}).")

if __name__ == '__main__':
    main()
