---
layout: page
title: Citation Analytics
permalink: /citations/
nav: true
nav_order: 99
---

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>

<style>
:root {
  --primary: #3b82f6;
  --bg: #ffffff;
  --bg-secondary: #f8fafc;
  --text: #1e293b;
  --text-muted: #64748b;
  --card: #ffffff;
  --border: #e2e8f0;
  --shadow: 0 1px 3px rgba(0,0,0,0.1);
}

@media (prefers-color-scheme: dark) {
  :root {
    --primary: #60a5fa;
    --bg: #0f172a;
    --bg-secondary: #1e293b;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --card: #1e293b;
    --border: #334155;
    --shadow: 0 1px 3px rgba(0,0,0,0.3);
  }
}

* { box-sizing: border-box; }

body {
  background: var(--bg);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  margin: 0;
  padding: 0;
}

.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 2rem;
}

.header {
  margin-bottom: 2rem;
}

.header h1 {
  margin: 0 0 0.5rem;
  font-size: 2rem;
  font-weight: 700;
}

.header p {
  margin: 0;
  color: var(--text-muted);
}

.papers-list {
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.paper-item {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 1.5rem;
  cursor: pointer;
  transition: all 0.2s ease;
  user-select: none;
}

.paper-item:hover {
  box-shadow: var(--shadow);
  transform: translateY(-1px);
}

.paper-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
}

.paper-info {
  flex: 1;
  min-width: 0;
}

.paper-title {
  margin: 0 0 0.5rem;
  font-size: 1.05rem;
  font-weight: 600;
  line-height: 1.4;
  word-break: break-word;
}

.paper-meta {
  display: flex;
  gap: 1rem;
  font-size: 0.85rem;
  color: var(--text-muted);
  flex-wrap: wrap;
}

.paper-stats {
  display: flex;
  align-items: center;
  gap: 1.5rem;
  white-space: nowrap;
}

.stat-badge {
  text-align: center;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: 700;
  color: var(--primary);
}

.stat-label {
  font-size: 0.75rem;
  color: var(--text-muted);
  margin-top: 0.25rem;
}

.expand-icon {
  font-size: 1.25rem;
  transition: transform 0.2s ease;
  flex-shrink: 0;
}

.paper-item.expanded .expand-icon {
  transform: rotate(180deg);
}

.paper-chart {
  display: none;
  margin-top: 1.5rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border);
}

.paper-item.expanded .paper-chart {
  display: block;
}

.chart-wrapper {
  position: relative;
  height: 300px;
  background: var(--bg-secondary);
  border-radius: 8px;
  padding: 1rem;
}

.loading {
  text-align: center;
  padding: 3rem 1rem;
  color: var(--text-muted);
}

.error {
  background: rgba(239, 68, 68, 0.1);
  border: 1px solid rgba(239, 68, 68, 0.3);
  border-radius: 8px;
  padding: 1rem;
  color: #fca5a5;
  text-align: center;
}

@media (max-width: 768px) {
  .container { padding: 1rem; }
  .header h1 { font-size: 1.5rem; }
  .paper-header { flex-direction: column; }
  .paper-stats { gap: 1rem; }
  .chart-wrapper { height: 250px; }
}
</style>

<div class="container">
  <div class="header">
    <h1>📊 Citation Analytics</h1>
    <p>Track daily citation changes for each paper</p>
  </div>

  <div class="papers-list" id="papersList">
    <div class="loading">Loading papers...</div>
  </div>
</div>

<script>
let allPapers = [];
let history = {};
let charts = {};

// 加载数据
async function loadData() {
  try {
    const [dataResp, historyResp] = await Promise.all([
      fetch('/data.json'),
      fetch('/papers_history.json')
    ]);
    
    if (!dataResp.ok || !historyResp.ok) throw new Error('Failed to load data');
    
    const data = await dataResp.json();
    history = await historyResp.json();
    
    allPapers = data.papers || [];
    
    if (allPapers.length === 0) {
      document.getElementById('papersList').innerHTML = '<div class="error">No papers found</div>';
      return;
    }
    
    renderPapers();
  } catch (err) {
    console.error('Error loading data:', err);
    document.getElementById('papersList').innerHTML = `<div class="error">Error loading data: ${err.message}</div>`;
  }
}

function renderPapers() {
  const container = document.getElementById('papersList');
  
  container.innerHTML = allPapers.map((paper, idx) => `
    <div class="paper-item" data-index="${idx}">
      <div class="paper-header">
        <div class="paper-info">
          <h3 class="paper-title">${escapeHtml(paper.title)}</h3>
          <div class="paper-meta">
            <span>${paper.year}</span>
            ${paper.venue ? `<span>${escapeHtml(paper.venue)}</span>` : ''}
          </div>
        </div>
        <div class="paper-stats">
          <div class="stat-badge">
            <div class="stat-value">${paper.citations}</div>
            <div class="stat-label">Citations</div>
          </div>
          <div class="expand-icon">▼</div>
        </div>
      </div>
      
      <div class="paper-chart">
        <div class="chart-wrapper">
          <canvas id="chart-${idx}"></canvas>
        </div>
      </div>
    </div>
  `).join('');
  
  // 添加点击事件
  document.querySelectorAll('.paper-item').forEach((item, idx) => {
    item.addEventListener('click', () => togglePaper(idx));
  });
}

function togglePaper(idx) {
  const item = document.querySelector(`[data-index="${idx}"]`);
  const isExpanded = item.classList.contains('expanded');
  
  // 关闭其他展开的项
  document.querySelectorAll('.paper-item.expanded').forEach(el => {
    if (el !== item) {
      el.classList.remove('expanded');
      const index = el.getAttribute('data-index');
      if (charts[index]) {
        charts[index].destroy();
        delete charts[index];
      }
    }
  });
  
  // 切换当前项
  item.classList.toggle('expanded');
  
  if (item.classList.contains('expanded') && !charts[idx]) {
    setTimeout(() => renderChart(idx), 100);
  } else if (!item.classList.contains('expanded') && charts[idx]) {
    charts[idx].destroy();
    delete charts[idx];
  }
}

function renderChart(idx) {
  const paper = allPapers[idx];
  const paperId = paper.id || paper.title;
  
  // 收集该论文的历史数据
  const dates = Object.keys(history).sort();
  const data = dates.map(date => {
    const dayData = history[date][paperId];
    return dayData ? dayData.citations : null;
  });
  
  const ctx = document.getElementById(`chart-${idx}`);
  if (!ctx) return;
  
  // 销毁旧图表
  if (charts[idx]) {
    charts[idx].destroy();
  }
  
  charts[idx] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        label: 'Citations',
        data: data,
        borderColor: 'rgb(59, 130, 246)',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 5,
        pointHoverRadius: 7,
        pointBackgroundColor: 'rgb(59, 130, 246)',
        pointBorderColor: '#fff',
        pointBorderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(0, 0, 0, 0.8)',
          padding: 12,
          titleFont: { size: 14 },
          bodyFont: { size: 13 }
        }
      },
      scales: {
        x: {
          grid: { display: false },
          ticks: { color: 'var(--text-muted)' }
        },
        y: {
          beginAtZero: true,
          grid: { color: 'rgba(0, 0, 0, 0.1)' },
          ticks: { color: 'var(--text-muted)' }
        }
      }
    }
  });
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

// 初始化
loadData();
</script>
