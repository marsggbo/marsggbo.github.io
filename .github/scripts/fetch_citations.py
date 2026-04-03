import csv
import json
import os
from datetime import datetime
from scholarly import scholarly

AUTHOR_ID = os.environ.get('SCHOLAR_USER_ID', 'LYNKm_8AAAAJ')
CSV_PATH = '_data/citations.csv'
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
    date_format_check = []
    
    # Check if CSV exists to decide on headers
    file_exists = os.path.exists(CSV_PATH)
    
    with open(CSV_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Date', 'Title', 'Citations'])
        
        for pub in author['publications']:
            title = pub['bib'].get('title', 'Unknown Title')
            citations = pub.get('num_citations', 0)
            writer.writerow([today, title, citations])

    print(f"Successfully appended today's ({today}) citations to {CSV_PATH}")

    # Now parse the entire CSV and build JSON
    history = []
    with open(CSV_PATH, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history.append(row)

    # Extract unique sorted dates
    dates = sorted(list(set([row['Date'] for row in history])))
    
    # Store mappings of Title -> {Date -> Citations}
    paper_map = {}
    
    for row in history:
        date = row['Date']
        title = row['Title']
        try:
            citations = int(row['Citations'])
        except ValueError:
            citations = 0
            
        if title not in paper_map:
            paper_map[title] = {}
        paper_map[title][date] = citations

    # Generate Chart.js optimized structure
    papers_data = []
    
    for title, dates_dict in paper_map.items():
        dataset = []
        last_known_citations = 0
        
        for d in dates:
            if d in dates_dict:
                last_known_citations = dates_dict[d]
            dataset.append(last_known_citations)
            
        papers_data.append({
            "title": title,
            "citations": dataset,
            "total_citations": dataset[-1] if dataset else 0
        })

    # Sort descending by latest citation count
    papers_data.sort(key=lambda x: x["total_citations"], reverse=True)

    json_data = {
        "dates": dates,
        "papers": papers_data
    }

    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    print(f"Successfully regenerated {JSON_PATH} with {len(papers_data)} papers.")

if __name__ == '__main__':
    main()
