#!/usr/bin/env python3
"""
生成可视化 HTML 页面
支持暗黑/白天模式自动切换
"""

import os
import json
import csv
from datetime import datetime
from pathlib import Path

OUTPUT_FILE = 'index.html'


def load_data():
    """加载历史数据和论文数据"""
    # 加载历史
    history = {}
    if Path('citations_history.csv').exists():
        with open('citations_history.csv', 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                history[row['date']] = row
    
    # 加载论文数据
    papers_data = {'papers': [], 'total_citations': 0, 'hindex': 0, 'i10index': 0}
    if Path('papers_data.json').exists():
        with open('papers_data.json', 'r', encoding='utf-8') as f:
            papers_data = json.load(f)
    
    return history, papers_data


def generate_html(history, papers_data):
    """生成 HTML 页面"""
    today = datetime.now().strftime('%Y-%m-%d')
    
    # 排序论文
    papers = papers_data.get('papers', [])
    papers.sort(key=lambda x: x.get('citations', 0), reverse=True)
    
    # 历史趋势数据
    dates = sorted(history.keys())
    total_trend = [(d, int(history[d]['total_citations'])) for d in dates]
    hindex_trend = [(d, int(history[d].get('hindex', 0) or 0)) for d in dates]
    
    # 计算变化
    if len(total_trend) >= 2:
        last_total = total_trend[-1][1]
        prev_total = total_trend[-2][1]
        total_change = last_total - prev_total
    else:
        total_change = 0
    
    html = f'''---
layout: page
title: Citation Analytics
permalink: /citations/
nav: true
nav_order: 99
---

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Citation Analytics - Xin He</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
    <style>
        /* 复用主题变量，自动适配暗黑/白天模式 */
        :root {{
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-card: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --primary: #3b82f6;
            --primary-hover: #2563eb;
            --success: #10b981;
            --warning: #f59e0b;
            --purple: #8b5cf6;
            --shadow: 0 1px 3px rgba(0,0,0,0.1);
            --shadow-lg: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        /* 暗黑模式 */
        @media (prefers-color-scheme: dark) {{
            :root {{
                --bg-primary: #0f172a;
                --bg-secondary: #1e293b;
                --bg-card: #1e293b;
                --text-primary: #f1f5f9;
                --text-secondary: #94a3b8;
                --text-muted: #64748b;
                --border-color: #334155;
                --shadow: 0 1px 3px rgba(0,0,0,0.3);
                --shadow-lg: 0 10px 40px rgba(0,0,0,0.4);
            }}
        }}
        
        body.theme-dark {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --text-primary: #f1f5f9;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border-color: #334155;
            --shadow: 0 1px 3px rgba(0,0,0,0.3);
            --shadow-lg: 0 10px 40px rgba(0,0,0,0.4);
        }}
        
        body.theme-light {{
            --bg-primary: #ffffff;
            --bg-secondary: #f8fafc;
            --bg-card: #ffffff;
            --text-primary: #1e293b;
            --text-secondary: #64748b;
            --text-muted: #94a3b8;
            --border-color: #e2e8f0;
            --shadow: 0 1px 3px rgba(0,0,0,0.1);
            --shadow-lg: 0 10px 40px rgba(0,0,0,0.1);
        }}
        
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: var(--bg-secondary);
            color: var(--text-primary);
            line-height: 1.6;
            padding: 2rem;
            transition: background 0.3s, color 0.3s;
        }}
        
        .container {{
            max-width: 1200px;
            margin: 0 auto;
        }}
        
        header {{
            margin-bottom: 2rem;
        }}
        
        h1 {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            margin-bottom: 0.5rem;
        }}
        
        .subtitle {{
            color: var(--text-secondary);
            font-size: 1rem;
        }}
        
        .back-link {{
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            color: var(--primary);
            text-decoration: none;
            font-size: 0.9rem;
            margin-bottom: 1rem;
            transition: color 0.2s;
        }}
        
        .back-link:hover {{
            color: var(--primary-hover);
        }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stat-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            box-shadow: var(--shadow);
            transition: transform 0.2s, box-shadow 0.2s, background 0.3s;
        }}
        
        .stat-card:hover {{
            transform: translateY(-2px);
            box-shadow: var(--shadow-lg);
        }}
        
        .stat-icon {{
            font-size: 1.5rem;
            margin-bottom: 0.5rem;
        }}
        
        .stat-value {{
            font-size: 2rem;
            font-weight: 700;
            color: var(--text-primary);
            line-height: 1.2;
        }}
        
        .stat-value.citations {{ color: var(--primary); }}
        .stat-value.hindex {{ color: var(--success); }}
        .stat-value.i10 {{ color: var(--warning); }}
        .stat-value.papers {{ color: var(--purple); }}
        
        .stat-change {{
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 0.25rem;
        }}
        
        .stat-change.positive {{ color: var(--success); }}
        .stat-change.negative {{ color: #ef4444; }}
        
        .stat-label {{
            color: var(--text-secondary);
            font-size: 0.85rem;
            margin-top: 0.5rem;
        }}
        
        .chart-container {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 1.5rem;
            margin-bottom: 2rem;
            box-shadow: var(--shadow);
            transition: background 0.3s, border-color 0.3s;
        }}
        
        .chart-title {{
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }}
        
        .papers-table {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            overflow: hidden;
            box-shadow: var(--shadow);
            transition: background 0.3s, border-color 0.3s;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        
        th, td {{
            padding: 1rem;
            text-align: left;
            border-bottom: 1px solid var(--border-color);
        }}
        
        th {{
            background: var(--bg-secondary);
            color: var(--text-secondary);
            font-weight: 600;
            font-size: 0.8rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        tr:last-child td {{ border-bottom: none; }}
        tr:hover td {{ background: var(--bg-secondary); }}
        
        .rank {{
            color: var(--text-muted);
            font-size: 0.9rem;
            width: 50px;
        }}
        
        .paper-title {{
            color: var(--text-primary);
            font-weight: 500;
        }}
        
        .paper-venue {{
            color: var(--text-muted);
            font-size: 0.85rem;
            margin-top: 0.25rem;
        }}
        
        .citation-count {{
            font-weight: 700;
            color: var(--primary);
            text-align: right;
            white-space: nowrap;
        }}
        
        .footer {{
            text-align: center;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid var(--border-color);
            color: var(--text-muted);
            font-size: 0.85rem;
        }}
        
        .footer a {{
            color: var(--primary);
            text-decoration: none;
        }}
        
        .footer a:hover {{
            text-decoration: underline;
        }}
        
        @media (max-width: 768px) {{
            body {{ padding: 1rem; }}
            h1 {{ font-size: 1.5rem; }}
            .stat-value {{ font-size: 1.5rem; }}
            .stats-grid {{ grid-template-columns: repeat(2, 1fr); }}
            th, td {{ padding: 0.75rem 0.5rem; }}
            .paper-title {{ font-size: 0.9rem; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <a href="/publications/" class="back-link">
            ← Back to Publications
        </a>
        
        <header>
            <h1>📊 Citation Analytics</h1>
            <p class="subtitle">Google Scholar • Updated {today}</p>
        </header>
        
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-icon">📈</div>
                <div class="stat-value citations">{papers_data.get('total_citations', 0):,}</div>
                <div class="stat-change {'positive' if total_change > 0 else 'negative' if total_change < 0 else ''}">
                    {'+' if total_change > 0 else ''}{total_change} from yesterday
                </div>
                <div class="stat-label">Total Citations</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">🏆</div>
                <div class="stat-value hindex">{papers_data.get('hindex', 0)}</div>
                <div class="stat-label">h-index</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📚</div>
                <div class="stat-value i10">{papers_data.get('i10index', 0)}</div>
                <div class="stat-label">i10-index</div>
            </div>
            <div class="stat-card">
                <div class="stat-icon">📄</div>
                <div class="stat-value papers">{len(papers)}</div>
                <div class="stat-label">Papers</div>
            </div>
        </div>
        
        <div class="chart-container">
            <h2 class="chart-title">📈 Citation Trend</h2>
            <canvas id="citationChart" style="max-height: 300px;"></canvas>
        </div>
        
        <div class="chart-container">
            <h2 class="chart-title">📚 Top 10 Papers</h2>
            <canvas id="papersChart" style="max-height: 400px;"></canvas>
        </div>
        
        <div class="papers-table">
            <table>
                <thead>
                    <tr>
                        <th class="rank">#</th>
                        <th>Paper</th>
                        <th style="text-align: right; width: 100px;">Citations</th>
                    </tr>
                </thead>
                <tbody>
'''
    
    # 添加论文列表
    for i, paper in enumerate(papers[:20], 1):
        title = paper.get('title', 'N/A')
        venue = paper.get('venue', '')
        citations = paper.get('citations', 0)
        
        html += f'''
                    <tr>
                        <td class="rank">{i}</td>
                        <td>
                            <div class="paper-title">{title}</div>
                            {f'<div class="paper-venue">{venue}</div>' if venue else ''}
                        </td>
                        <td class="citation-count">{citations:,}</td>
                    </tr>
'''
    
    html += '''
                </tbody>
            </table>
        </div>
        
        <footer class="footer">
            <p>Generated by GitHub Actions • Data from <a href="https://scholar.google.com" target="_blank">Google Scholar</a></p>
            <p>Updated at ''' + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + '''</p>
        </footer>
    </div>
    
    <script>
        // 检测主题偏好并应用
        function applyTheme() {
            const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
            const isDarkTheme = document.body.classList.contains('theme-dark') || 
                              (!document.body.classList.contains('theme-light') && prefersDark);
            return isDarkTheme;
        }
        
        const isDark = applyTheme();
        const gridColor = isDark ? 'rgba(255,255,255,0.1)' : 'rgba(0,0,0,0.1)';
        const textColor = isDark ? '#94a3b8' : '#64748b';
        const titleColor = isDark ? '#f1f5f9' : '#1e293b';
        
        // 历史数据
        const totalTrend = ''' + json.dumps([{"date": d, "total": t} for d, t in total_trend]) + ''';
        const hindexTrend = ''' + json.dumps([{"date": d, "hindex": h} for d, h in hindex_trend]) + ''';
        const topPapers = ''' + json.dumps(papers[:10]) + ''';
        
        // 格式化日期
        const formatDate = (dateStr) => {
            const date = new Date(dateStr);
            return date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
        };
        
        // 总引用趋势图
        if (totalTrend.length > 0) {
            new Chart(document.getElementById('citationChart'), {
                type: 'line',
                data: {
                    labels: totalTrend.map(d => formatDate(d.date)),
                    datasets: [{
                        label: 'Total Citations',
                        data: totalTrend.map(d => d.total),
                        borderColor: '#3b82f6',
                        backgroundColor: 'rgba(59, 130, 246, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 4,
                        pointHoverRadius: 6,
                        pointBackgroundColor: '#3b82f6'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: isDark ? 'rgba(30,41,59,0.95)' : 'rgba(255,255,255,0.95)',
                            titleColor: titleColor,
                            bodyColor: textColor,
                            borderColor: gridColor,
                            borderWidth: 1,
                            padding: 12,
                            displayColors: false
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: gridColor },
                            ticks: { color: textColor }
                        },
                        y: {
                            grid: { color: gridColor },
                            ticks: { color: textColor }
                        }
                    }
                }
            });
        }
        
        // 论文引用柱状图
        if (topPapers.length > 0) {
            const sorted = [...topPapers].sort((a, b) => b.citations - a.citations);
            
            new Chart(document.getElementById('papersChart'), {
                type: 'bar',
                data: {
                    labels: sorted.map(p => p.title.length > 35 ? p.title.substring(0, 35) + '...' : p.title),
                    datasets: [{
                        label: 'Citations',
                        data: sorted.map(p => p.citations),
                        backgroundColor: [
                            'rgba(59, 130, 246, 0.8)',
                            'rgba(16, 185, 129, 0.8)',
                            'rgba(245, 158, 11, 0.8)',
                            'rgba(139, 92, 246, 0.8)',
                            'rgba(239, 68, 68, 0.8)',
                            'rgba(8, 145, 178, 0.8)',
                            'rgba(236, 72, 153, 0.8)',
                            'rgba(132, 204, 22, 0.8)',
                            'rgba(14, 165, 233, 0.8)',
                            'rgba(251, 146, 60, 0.8)'
                        ],
                        borderRadius: 6
                    }]
                },
                options: {
                    indexAxis: 'y',
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            backgroundColor: isDark ? 'rgba(30,41,59,0.95)' : 'rgba(255,255,255,0.95)',
                            titleColor: titleColor,
                            bodyColor: textColor,
                            borderColor: gridColor,
                            borderWidth: 1,
                            padding: 12
                        }
                    },
                    scales: {
                        x: {
                            grid: { color: gridColor },
                            ticks: { color: textColor }
                        },
                        y: {
                            grid: { display: false },
                            ticks: { color: titleColor, font: { size: 11 } }
                        }
                    }
                }
            });
        }
        
        // 监听主题变化
        window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
            location.reload();
        });
    </script>
</body>
</html>'''
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ 生成了 {OUTPUT_FILE}")


def main():
    print("=" * 50)
    print("📊 生成可视化页面")
    print("=" * 50)
    
    history, papers_data = load_data()
    
    if not history:
        print("⚠️  没有历史数据先生成")
        return
    
    print(f"📈 有 {len(history)} 天的历史数据")
    print(f"📚 有 {len(papers_data.get('papers', []))} 篇论文数据")
    
    generate_html(history, papers_data)
    
    print()
    print("✅ 完成!")


if __name__ == '__main__':
    main()
