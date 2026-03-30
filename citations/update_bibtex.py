#!/usr/bin/env python3
"""
更新 papers.bib 文件
如果 Google Scholar 有新论文但 bib 文件中没有，则添加
"""

import os
import re
import json
from pathlib import Path
from datetime import datetime

BIB_FILE = '../_bibliography/papers.bib'  # 相对于 citations 目录
PAPERS_FILE = 'papers_data.json'

# 从 papers.bib 中提取已有的论文标题
def extract_existing_titles():
    """从现有的 BibTeX 文件中提取论文标题"""
    titles = set()
    
    if not Path(BIB_FILE).exists():
        return titles
    
    with open(BIB_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 匹配 title 字段
    pattern = r'title\s*=\s*\{([^}]+)\}'
    matches = re.findall(pattern, content, re.IGNORECASE)
    
    for title in matches:
        # 标准化：转小写，去除标点
        normalized = re.sub(r'[^\w\s]', '', title.lower()).strip()
        titles.add(normalized)
    
    return titles


def normalize_title(title):
    """标准化论文标题"""
    return re.sub(r'[^\w\s]', '', title.lower()).strip()


def generate_bibtex_entry(paper):
    """从论文数据生成 BibTeX 条目"""
    title = paper['title']
    citations = paper['citations']
    year = paper.get('year', datetime.now().year)
    scholar_id = paper.get('scholar_id', '')
    
    # 生成 cite key
    first_author = paper.get('author', 'He, Xin').split(' and ')[0].split(',')[0]
    first_author_last = first_author.split()[-1].lower() if first_author else 'he'
    cite_key = f"{first_author_last}{year}"
    
    # 判断类型
    venue = paper.get('venue', '').lower()
    if 'arxiv' in venue or 'preprint' in venue:
        bib_type = 'article'
    elif 'conference' in venue or 'proceedings' in venue:
        bib_type = 'inproceedings'
    else:
        bib_type = 'article'
    
    # 构建 BibTeX
    lines = [
        f"@{bib_type}{{{cite_key},",
        f"  title={{{title}}},",
        f"  author={{{paper.get('author', 'He, Xin')}}},",
        f"  year={{{year}}},",
        f"  google_scholar_id={{{scholar_id}}},",
        f"  citations={{{citations}}},",
        f"  bibtex_show={{true}},",
        f"  abbr={{GS}}",
        "}",
    ]
    
    return '\n'.join(lines)


def update_bibtex():
    """检查并更新 BibTeX 文件"""
    print("=" * 50)
    print("🔍 检查是否有新论文需要添加...")
    print("=" * 50)
    
    # 读取当前论文数据
    if not Path(PAPERS_FILE).exists():
        print("⚠️  未找到 papers_data.json，跳过 BibTeX 更新")
        return
    
    with open(PAPERS_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    papers = data.get('papers', [])
    print(f"📚 Google Scholar 上有 {len(papers)} 篇论文")
    
    # 获取已有的标题
    existing_titles = extract_existing_titles()
    print(f"📁 papers.bib 中已有 {len(existing_titles)} 篇论文")
    
    # 找出新论文
    new_papers = []
    for paper in papers:
        title = paper.get('title', '')
        normalized = normalize_title(title)
        
        # 检查是否已存在
        if normalized and normalized not in existing_titles:
            # 还需要检查部分匹配（避免标题略有不同的情况）
            is_new = True
            for existing in existing_titles:
                # 如果有超过 80% 的字符匹配，认为是同一篇
                if len(normalized) > 10 and len(existing) > 10:
                    common_chars = set(normalized) & set(existing)
                    if len(common_chars) / max(len(normalized), len(existing)) > 0.8:
                        is_new = False
                        break
            
            if is_new:
                new_papers.append(paper)
    
    if not new_papers:
        print()
        print("✅ 没有发现新论文，BibTeX 无需更新")
        return
    
    print()
    print(f"🆕 发现 {len(new_papers)} 篇新论文!")
    print()
    
    # 生成新的 BibTeX 条目
    new_entries = []
    for paper in new_papers:
        entry = generate_bibtex_entry(paper)
        new_entries.append(entry)
        print(f"   + {paper['title'][:60]}...")
    
    # 追加到 bib 文件
    bib_path = Path(BIB_FILE)
    
    if bib_path.exists():
        # 读取现有内容
        with open(bib_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 在文件末尾添加新条目
        new_content = content.rstrip() + '\n\n' + '\n\n'.join(new_entries) + '\n'
    else:
        # 创建新文件
        new_content = '\n\n'.join(new_entries) + '\n'
        # 确保目录存在
        bib_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 写入
    with open(bib_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print()
    print(f"✅ 已更新 {BIB_FILE}，添加了 {len(new_papers)} 篇论文")
    
    # 列出修改
    print()
    print("新增的论文:")
    for i, paper in enumerate(new_papers, 1):
        print(f"  {i}. {paper['title']}")


if __name__ == '__main__':
    update_bibtex()
