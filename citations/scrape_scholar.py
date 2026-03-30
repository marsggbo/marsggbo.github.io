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
    
    # 搜索并填充作者信息
    search_results = scholarly.search_author_id(SCHOLAR_USER_ID)
    author = scholarly.fill(search_results)
    
    logger.info(f"找到作者: {author.get('name', 'N/A')}")
    logger.info(f"机构: {author.get('affiliation', 'N/A')}")
    
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
    """保存最新论文数据"""
    with open(PAPERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"保存了 {len(data['papers'])} 篇论文数据")


def main():
    print("=" * 50)
    logger.info("开始抓取 Google Scholar")
    print("=" * 50)
    
    data = None
    
    # 方法: scholarly
    try:
        data = scrape_with_scholarly()
        logger.info("✅ scholarly 抓取成功")
    except Exception as e:
        logger.error(f"❌ scholarly 失败: {e}")
        import traceback
        traceback.print_exc()
    
    if data is None:
        logger.error("抓取失败")
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
