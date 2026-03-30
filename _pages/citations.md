---
layout: page
title: Citation Analytics
permalink: /citations/
nav: true
nav_order: 99
---

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
:root { 
  --primary: #3b82f6; 
  --bg: #f8fafc; 
  --text: #1e293b; 
  --card: #fff;
  --border: #e2e8f0;
}
@media (prefers-color-scheme: dark) { 
  :root { 
    --bg: #0f172a; 
    --text: #f1f5f9; 
    --card: #1e293b;
    --border: #334155;
  } 
}

body { 
  background: var(--bg); 
  color: var(--text); 
  font-family: system-ui; 
  padding: 2rem; 
  max-width: 1400px; 
  margin: 0 auto; 
}

.header {
  margin-bottom: 2rem;
}

.header h1 {
  margin: 0 0 0.5rem;
  font-size: 2rem;
}

.header p {
  margin: 0;
  opacity: 0.7;
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
  transition: all 0.2s;
}

.paper-item:hover {
  box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  transform: translateY(-2px);
}

.paper-header {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 1rem;
}

.paper-title {
  flex: 1;
  font-weight: 600;
  font-size: 1.05rem;
  margin: 0;
}

.paper-meta {
  display: flex;
  gap: 1rem;
  margin-top: 0.5rem;
  font-size: 0.85rem;
  opacity: 0.7;
}

.paper-stats {
  display: flex;
  gap: 1.5rem;
  align-items: center;
}

.stat-badge {
  text-align: center;
}

.stat-value {
  font-size: 1.5rem;
  font-weight: bold;
  color: var(--primary);
}

.stat-label {
  font-size: 0.75rem;
  opacity: 0.7;
  margin-top: 0.25rem;
}

.expand-icon {
  font-size: 1.5rem;
  transition: transform 0.2s;
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

.chart-container {
  position: relative;
  height: 250px;
}

.loading {
  text-align: center;
  padding: 2rem;
  opacity: 0.5;
}

.empty {
  text-align: center;
  padding: 2rem;
  opacity: 0.5;
}
</style>

<div class="header">
  <h1>📊 Citation Analytics</h1>
  <p>Track daily citation changes for each paper</p>
</div>

<div class="papers-list" id="papersList">
  <div class="loading">Loading papers...</div>
</div>

<script>
let allPapers = [];
let history = {};
let charts = {};

// 加载数据
Promise.all([
  fetch('/data.json').then(r => r.json()).catch(() => ({papers: []})),
  fetch('/papers_history.json').then(r => r.json()).catch(() => ({}))
]).then(([data, hist]) => {
  allPapers = data.papers || [];
  history = hist || {};
  
  if (allPapers.length === 0) {
    document.getElementById('papersList').innerHTML = '<div class="empty">No papers found</div>';
    return;
  }
  
  renderPapers();
});

function renderPapers() {
  const container = document.getElementById('papersList');
  container.innerHTML = allPapers.map((paper, idx) => `
    <div class="paper-item" data-index="${idx}">
      <div class="paper-header">
        <div style="flex: 1;">
          <h3 class="paper-title">${paper.title}</h3>
          <div class="paper-meta">
            <span>${paper.year}</span>
            ${paper.venue ? `<span>${paper.venue}</span>` : ''}
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
        <div class="chart-container">
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
  item.classList.toggle('expanded');
  
  if (item.classList.contains('expanded') && !charts[idx]) {
    setTimeout(() => renderChart(idx), 100);
  }
}

function renderChart(idx) {
  const paper = allPapers[idx];
  const paperId = paper.id || paper.title;
  
  // 收集该论文的历史数据
  const dates = Object.keys(history).sort();
  const data = dates.map(date => {
    const dayData = history[date][paperId];
    return dayData ? dayData.citations : 0;
  });
  
  const ctx = document.getElementById(`chart-${idx}`);
  if (!ctx) return;
  
  charts[idx] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        label: 'Citations',
        data: data,
        borderColor: '#3b82f6',
        backgroundColor: 'rgba(59, 130, 246, 0.1)',
        fill: true,
        tension: 0.3,
        pointRadius: 4,
        pointHoverRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false }
      },
      scales: {
        x: { grid: { display: false } },
        y: { beginAtZero: true }
      }
    }
  });
}
</script>
