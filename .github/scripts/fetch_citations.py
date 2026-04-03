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
    else:
        data = {"dates": [], "papers": []}
    
    # Update dates list if today is new
    if today not in data["dates"]:
        data["dates"].append(today)
    
    today_index = data["dates"].index(today)
    
    # Create mapping from existing papers
    paper_map = {p["title"]: p for p in data["papers"]}
    
    # Sync new data
    for pub in author['publications']:
        title = pub['bib'].get('title', 'Unknown Title')
        citations = pub.get('num_citations', 0)
        
        if title not in paper_map:
            # New paper discovered
            paper_map[title] = {
                "title": title,
                "citations": [0] * len(data["dates"]),
                "total_citations": citations
            }
        
        # Ensure the citations array matches the length of dates
        c_list = paper_map[title]["citations"]
        while len(c_list) < len(data["dates"]):
            c_list.append(c_list[-1] if c_list else 0)
            
        # Update today's entry
        c_list[today_index] = citations
        paper_map[title]["total_citations"] = citations

    # Update all datasets to ensure consistent lengths and carry over citations for un-updated papers
    for p in paper_map.values():
        while len(p["citations"]) < len(data["dates"]):
            p["citations"].append(p["citations"][-1] if p["citations"] else 0)

    # Convert back to list and sort
    papers_list = list(paper_map.values())
    papers_list.sort(key=lambda x: x["total_citations"], reverse=True)
    
    data["papers"] = papers_list

    # Save format
    with open(JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Successfully generated {JSON_PATH} with {len(papers_list)} papers (Date: {today}).")

if __name__ == '__main__':
    main()
