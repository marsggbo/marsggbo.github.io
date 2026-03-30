#!/usr/bin/env python3
"""
Google Scholar Citation Scraper
无头浏览器方式抓取，兼容 GitHub Actions
"""

import os
import re
import json
import csv
from datetime import datetime
from pathlib import Path

try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False

# 配置
SCHOLAR_USER_ID = os.environ.get('SCHOLAR_USER_ID', 'LYNKm_8AAAAJ')
SCHOLAR_URL = f"https://scholar.google.com/citations?user={SCHOLAR_USER_ID}"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}


def scrape_with_requests():
    """使用 requests + BeautifulSoup 抓取"""
    import requests
    
    session = requests.Session()
    response = session.get(SCHOLAR_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    
    soup = BeautifulSoup(response.text, 'html.parser')
    return parse_soup(soup)


def scrape_with_httpx():
    """使用 httpx 抓取（备用）"""
    with httpx.Client(headers=HEADERS, timeout=30) as client:
        response = client.get(SCHOLAR_URL)
        soup = BeautifulSoup(response.text, 'html.parser')
        return parse_soup(soup)


def scrape_with_urllib():
    """纯 urllib 实现（最后备用）"""
    from urllib.request import Request, urlopen
    
    req = Request(SCHOLAR_URL, headers=HEADERS)
    with urlopen(req, timeout=30) as response:
        html = response.read().decode('utf-8')
    
    soup = BeautifulSoup(html, 'html.parser')
    return parse_soup(soup)


def parse_soup(soup):
    """解析 BeautifulSoup 对象"""
    papers = []
    
    # 方法1: 解析论文列表（gsc_a_b 容器）
    for item in soup.select('#gsc_a_b .gsc_a_t'):
        title_elem = item.select_one('.gsc_a_at')
        authors_elem = item.select_one('.gs_gray')
        cites_elem = item.select_one('.gsc_a_c-n')
        
        if title_elem:
            title = title_elem.get_text(strip=True)
            # 获取链接中的论文ID
            link = title_elem.get('href', '')
            authors = authors_elem.get_text(strip=True) if authors_elem else ""
            cites_text = cites_elem.get_text(strip=True) if cites_elem else "0"
            
            # 提取引用数
            citations = int(re.sub(r'[^\d]', '', cites_text) or 0)
            
            papers.append({
                'title': title,
                'authors': authors,
                'citations': citations,
                'year': ''
            })
    
    # 如果没找到，尝试方法2
    if not papers:
        for item in soup.select('.gsc_a_tr'):
            title_elem = item.select_one('.gsc_a_at')
            cites_elem = item.select_one('.gsc_a_c')
            year_elem = item.select_one('.gsc_a_y')
            
            if title_elem:
                papers.append({
                    'title': title_elem.get_text(strip=True),
                    'authors': '',
                    'citations': int(re.sub(r'[^\d]', '', cites_elem.get_text(strip=True) if cites_elem else '0') or 0),
                    'year': year_elem.get_text(strip=True) if year_elem else ''
                })
    
    # 获取总引用数
    total_citations = 0
    total_elem = soup.select_one('#gsc_rsb_stm')
    if total_elem:
        total_citations = int(re.sub(r'[^\d]', '', total_elem.get_text(strip=True)) or 0)
    
    # h-index, i10-index
    h_index = 0
    i10_index = 0
    
    for stat in soup.select('#gsc_rsb_sts .gsc_rsb_std'):
        text = stat.get_text(strip=True)
        if text.isdigit():
            if h_index == 0:
                h_index = int(text)
            elif i10_index == 0:
                i10_index = int(text)
                break
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'total_citations': total_citations,
        'h_index': h_index,
        'i10_index': i10_index,
        'papers': papers
    }


def load_history():
    """加载历史数据"""
    history_file = Path('citations_history.csv')
    if not history_file.exists():
        return {}
    
    history = {}
    with open(history_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history[row['date']] = row
    return history


def save_history(history, data):
    """保存历史数据"""
    # 更新当前日期
    history[data['date']] = {
        'date': data['date'],
        'total_citations': data['total_citations'],
        'h_index': data.get('h_index', 0),
        'i10_index': data.get('i10_index', 0),
        'papers_count': len(data['papers'])
    }
    
    with open('citations_history.csv', 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'total_citations', 'h_index', 'i10_index', 'papers_count'])
        writer.writeheader()
        for entry in sorted(history.values(), key=lambda x: x['date']):
            writer.writerow(entry)
    
    print(f"✅ 保存了 {len(history)} 天的历史数据")


def generate_visualization(data, history):
    """生成可视化 HTML"""
    today = data['date']
    
    # 准备时间序列数据
    dates = sorted(history.keys())
    
    # 总引用趋势
    total_trend = [(d, int(history[d]['total_citations'])) for d in dates]
    
    # 论文数量趋势
    papers_trend = [(d, int(history[d].get('papers_count', 0))) for d in dates]
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Citation Tracker - {SCHOLAR_USER_ID}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        :root {{
            --primary: #2563eb;
            --secondary: #7c3aed;
            --bg: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
            --card-bg: rgba(30, 41, 59, 0.8);
            --text: #f1f5f9;
            --text-muted: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 2rem;
            backdrop-filter: blur(10px);
        }}
        
        .container {{ max-width: 1200px; margin: 0 auto; }}
        
        header {{
            text-align: center;
            margin-bottom: 3rem;
        }}
        
        h1 {{
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }}
        
        .subtitle {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.1);
            transition: transform 0.3s, box-shadow 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-4px);
            box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        }}
        
        .stat-icon {{ font-size: 2rem; margin-bottom: 0.5rem; }}
        
        .stat-value {{
            font-size: 2.5rem;
            font-weight: bold;
            color: var(--primary);
        }}
        
        .stat-value.citations {{ color: var(--success); }}
        .stat-value.h-index {{ color: var(--warning); }}
        .stat-value.papers {{ color: var(--secondary); }}
        
        .stat-label {{
            color: var(--text-muted);
            margin-top: 0.5rem;
            font-size: 0.9rem;
        }}
        
        .chart-container {{
            background: var(--card-bg);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        .chart-title {{
            font-size: 1.25rem;
            margin-bottom: 1.5rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .papers-table {{
            background: var(--card-bg);
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        
        table {{ width: 100%; border-collapse: collapse; }}
        
        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        
        th {{
            background: rgba(255,255,255,0.05);
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        tr:hover {{ background: rgba(255,255,255,0.02); }}
        tr:last-child td {{ border-bottom: none; }}
        
        .citation-count {{
            font-weight: bold;
            color: var(--success);
            font-size: 1.1rem;
        }}
        
        .paper-rank {{
            color: var(--text-muted);
            font-size: 0.9rem;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 3rem;
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
        
        .footer a {{
            color: var(--primary);
            text-decoration: none;
        }}
        
        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            h1 {{ font-size: 1.8rem; }}
            .stat-value {{ font-size: 2rem; }}
            th, td {{ padding: 0.75rem 0.5rem; font-size: 0.85rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Citation Tracker</h1>
            <p class="subtitle">Google Scholar • Updated {today}</p>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">📈</div>
                <div class="stat-value citations">{data['total_citations']:,}</div>
                <div class="stat-label">Total Citations</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🏆</div>
                <div class="stat-value h-index">{data.get('h_index', '-')}</div>
                <div class="stat-label">h-index</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📄</div>
                <div class="stat-value papers">{len(data['papers'])}</div>
                <div class="stat-label">Papers</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📅</div>
                <div class="stat-value" style="font-size: 1.5rem;">{len(history)}</div>
                <div class="stat-label">Days Tracked</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2 class="chart-title">📈 Citation History</h2>
            <canvas id="citationChart" height="300"></canvas>
        </div>
        
        <div class="chart-container">
            <h2 class="chart-title">📚 Top 10 Papers by Citations</h2>
            <canvas id="papersChart" height="300"></canvas>
        </div>
        
        <div class="papers-table">
            <table>
                <thead>
                    <tr>
                        <th style="width: 60px;">#</th>
                        <th>Title</th>
                        <th style="width: 100px;">Citations</th>
                    </tr>
                </thead>
                <tbody>
'''
    
    # 添加论文列表
    sorted_papers = sorted(data['papers'], key=lambda x: x['citations'], reverse=True)
    for i, paper in enumerate(sorted_papers[:20], 1):
        html += f'''
                    <tr>
                        <td class="paper-rank">{i}</td>
                        <td>{paper['title']}</td>
                        <td class="citation-count">{paper['citations']:,}</td>
                    </tr>
'''
    
    html += '''
                </tbody>
            </table>
        </div>
        
        <div class="footer">
            <p>Automated by GitHub Actions • Data from <a href="https://scholar.google.com" target="_blank">Google Scholar</a></p>
            <p style="margin-top: 0.5rem;">Generated at ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
        </div>
    </div>
    
    <script>
        // 历史数据
        const totalTrend = ''' + json.dumps([{"date": d, "total": t} for d, t in total_trend]) + ''';
        const papersTrend = ''' + json.dumps(data['papers']) + ''';
        
        // 总引用趋势图
        new Chart(document.getElementById('citationChart'), {
            type: 'line',
            data: {
                labels: totalTrend.map(d => d.date),
                datasets: [{
                    label: 'Total Citations',
                    data: totalTrend.map(d => d.total),
                    borderColor: '#2563eb',
                    backgroundColor: 'rgba(37, 99, 235, 0.1)',
                    fill: true,
                    tension: 0.4,
                    pointRadius: 4,
                    pointHoverRadius: 6
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.9)',
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8',
                        borderColor: 'rgba(255,255,255,0.1)',
                        borderWidth: 1
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    }
                }
            }
        });
        
        // 论文引用柱状图
        const topPapers = papersTrend
            .sort((a, b) => b.citations - a.citations)
            .slice(0, 10);
            
        new Chart(document.getElementById('papersChart'), {
            type: 'bar',
            data: {
                labels: topPapers.map(p => p.title.length > 30 ? p.title.substring(0, 30) + '...' : p.title),
                datasets: [{
                    label: 'Citations',
                    data: topPapers.map(p => p.citations),
                    backgroundColor: [
                        'rgba(37, 99, 235, 0.8)',
                        'rgba(124, 58, 237, 0.8)',
                        'rgba(16, 185, 129, 0.8)',
                        'rgba(245, 158, 11, 0.8)',
                        'rgba(239, 68, 68, 0.8)',
                        'rgba(8, 145, 178, 0.8)',
                        'rgba(79, 70, 229, 0.8)',
                        'rgba(132, 204, 22, 0.8)',
                        'rgba(236, 72, 153, 0.8)',
                        'rgba(14, 165, 233, 0.8)'
                    ],
                    borderRadius: 8
                }]
            },
            options: {
                indexAxis: 'y',
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        backgroundColor: 'rgba(30, 41, 59, 0.9)',
                        titleColor: '#f1f5f9',
                        bodyColor: '#94a3b8'
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255,255,255,0.05)' },
                        ticks: { color: '#94a3b8' }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#f1f5f9', font: { size: 11 } }
                    }
                }
            }
        });
    </script>
</body>
</html>'''
    
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ 生成了可视化页面 index.html")


def main():
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始抓取 Google Scholar...")
    print(f"用户ID: {SCHOLAR_USER_ID}")
    
    # 尝试不同的抓取方法
    data = None
    errors = []
    
    # 方法1: requests + BeautifulSoup
    if HAS_BS4:
        try:
            import requests
            data = scrape_with_requests()
            print("✅ 使用 requests + BeautifulSoup 抓取成功")
        except Exception as e:
            errors.append(f"requests: {e}")
    
    # 方法2: httpx + BeautifulSoup
    if data is None and HAS_HTTPX:
        try:
            data = scrape_with_httpx()
            print("✅ 使用 httpx + BeautifulSoup 抓取成功")
        except Exception as e:
            errors.append(f"httpx: {e}")
    
    # 方法3: 纯 urllib
    if data is None:
        try:
            data = scrape_with_urllib()
            print("✅ 使用 urllib + BeautifulSoup 抓取成功")
        except Exception as e:
            errors.append(f"urllib: {e}")
    
    if data is None:
        print(f"❌ 所有抓取方法都失败了:")
        for err in errors:
            print(f"   - {err}")
        exit(1)
    
    print(f"\n📊 抓取结果:")
    print(f"   总引用数: {data['total_citations']}")
    print(f"   h-index: {data.get('h_index', 'N/A')}")
    print(f"   论文数量: {len(data['papers'])}")
    
    if data['papers']:
        print(f"\n📚 Top 5 论文:")
        for i, paper in enumerate(sorted(data['papers'], key=lambda x: x['citations'], reverse=True)[:5], 1):
            print(f"   {i}. {paper['title'][:50]}... ({paper['citations']} citations)")
    
    # 加载并更新历史
    history = load_history()
    save_history(history, data)
    
    # 生成可视化
    generate_visualization(data, history)
    
    print(f"\n🎉 完成！")


if __name__ == '__main__':
    main()
