#!/usr/bin/env python3
"""
Google Scholar Citation Scraper
使用 scholarly 库抓取论文引用数据
"""

import os
import re
import json
import csv
import time
import logging
from datetime import datetime
from pathlib import Path

# 配置
SCHOLAR_USER_ID = os.environ.get('SCHOLAR_USER_ID', 'LYNKm_8AAAAJ')
HISTORY_FILE = 'citations_history.csv'
PAPERS_FILE = 'papers_data.json'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def scrape_with_scholarly():
    """使用 scholarly 库抓取"""
    from scholarly import scholarly
    
    logger.info(f"正在抓取 Google Scholar: {SCHOLAR_USER_ID}")
    
    # 搜索作者
    search_results = list(scholarly.search_author_id(SCHOLAR_USER_ID))
    if not search_results:
        raise Exception("未找到该作者")
    
    author = search_results[0]
    logger.info(f"找到作者: {author.get('name', 'N/A')}")
    
    # 填充作者详情（包括所有出版物）
    author = scholarly.fill(author)
    
    papers = []
    for pub in author.get('publications', []):
        bib = pub.get('bib', {})
        papers.append({
            'title': bib.get('title', 'N/A'),
            'author': bib.get('author', 'N/A'),
            'venue': bib.get('venue', ''),
            'year': bib.get('pub_year', ''),
            'citations': pub.get('num_citations', 0),
            'scholar_id': pub.get('scholar_id', ''),
            'gsrank': pub.get('gsrank', 0),
        })
    
    # 按引用数排序
    papers.sort(key=lambda x: x['citations'], reverse=True)
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'time': datetime.now().strftime('%H:%M:%S'),
        'author_name': author.get('name', ''),
        'affiliation': author.get('affiliation', ''),
        'total_citations': author.get('citedby', 0),
        'hindex': author.get('hindex', 0),
        'i10index': author.get('i10index', 0),
        'papers': papers
    }


def scrape_fallback():
    """备用抓取方法：使用 requests + BeautifulSoup"""
    import requests
    from bs4 import BeautifulSoup
    
    logger.info("使用备用方法抓取...")
    
    url = f"https://scholar.google.com/citations?user={SCHOLAR_USER_ID}&view_op=list_works&sortby=pubdate"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    response = requests.get(url, headers=headers, timeout=30)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    papers = []
    for item in soup.select('#gsc_a_b .gsc_a_t'):
        title_elem = item.select_one('.gsc_a_at')
        cites_elem = item.select_one('.gsc_a_c-n')
        
        if title_elem:
            title = title_elem.get_text(strip=True)
            citations = int(re.sub(r'[^\d]', '', cites_elem.get_text(strip=True) if cites_elem else '0') or 0)
            papers.append({
                'title': title,
                'citations': citations,
                'scholar_id': ''
            })
    
    return {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_citations': sum(p['citations'] for p in papers),
        'papers': papers
    }


def load_history():
    """加载历史数据"""
    if not Path(HISTORY_FILE).exists():
        return {}
    
    history = {}
    with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            history[row['date']] = row
    return history


def save_history(history, data):
    """保存历史数据"""
    history[data['date']] = {
        'date': data['date'],
        'total_citations': data['total_citations'],
        'hindex': data.get('hindex', ''),
        'i10index': data.get('i10index', ''),
        'papers_count': len(data['papers'])
    }
    
    with open(HISTORY_FILE, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'total_citations', 'hindex', 'i10index', 'papers_count'])
        writer.writeheader()
        for entry in sorted(history.values(), key=lambda x: x['date']):
            writer.writerow(entry)
    
    logger.info(f"保存了 {len(history)} 天的历史数据")


def save_papers(data):
    """保存最新论文数据（用于 BibTeX 更新和可视化）"""
    with open(PAPERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"保存了 {len(data['papers'])} 篇论文数据")


def main():
    print("=" * 50)
    logger.info("开始抓取 Google Scholar")
    print("=" * 50)
    
    data = None
    error_msg = None
    
    # 方法1: scholarly
    try:
        data = scrape_with_scholarly()
        logger.info("✅ scholarly 抓取成功")
    except Exception as e:
        error_msg = f"scholarly: {e}"
        logger.warning(f"❌ scholarly 失败: {e}")
    
    # 方法2: 备用
    if data is None:
        try:
            data = scrape_fallback()
            logger.info("✅ 备用方法抓取成功")
        except Exception as e:
            error_msg += f", fallback: {e}"
            logger.error(f"❌ 备用方法也失败: {e}")
    
    if data is None:
        logger.error("所有抓取方法都失败了")
        # 写入空数据标记
        with open('error.log', 'a') as f:
            f.write(f"{datetime.now()} - {error_msg}\n")
        return
    
    # 打印结果
    print()
    logger.info(f"📊 抓取结果:")
    logger.info(f"   作者: {data.get('author_name', 'N/A')}")
    logger.info(f"   总引用数: {data['total_citations']}")
    logger.info(f"   h-index: {data.get('hindex', 'N/A')}")
    logger.info(f"   i10-index: {data.get('i10index', 'N/A')}")
    logger.info(f"   论文数量: {len(data['papers'])}")
    
    if data['papers']:
        print()
        logger.info("📚 Top 5 论文:")
        for i, paper in enumerate(data['papers'][:5], 1):
            logger.info(f"   {i}. {paper['title'][:60]}... ({paper['citations']} citations)")
    
    # 保存数据
    save_history(load_history(), data)
    save_papers(data)
    
    print()
    logger.info("✅ 完成!")


if __name__ == '__main__':
    main()
