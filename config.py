# 配置 - Scholar ID
SCHOLAR_URL = "https://scholar.google.com/citations?user=LYNKm_8AAAAJ"

def get_scholar_id():
    url = SCHOLAR_URL.strip()
    if 'user=' in url:
        return url.split('user=')[1].split('&')[0]
    return url.strip()
