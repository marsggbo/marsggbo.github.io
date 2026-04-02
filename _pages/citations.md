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
  --primary: #6366f1;
  --primary-light: #818cf8;
  --success: #22c55e;
  --warning: #f59e0b;
  --bg: #ffffff;
  --bg-secondary: #f8fafc;
  --bg-card: #ffffff;
  --text: #1e293b;
  --text-muted: #64748b;
  --border: #e2e8f0;
  --shadow: 0 1px 3px rgba(0,0,0,0.08);
  --shadow-lg: 0 10px 40px rgba(0,0,0,0.12);
  --radius: 16px;
  --radius-sm: 10px;
}

@media (prefers-color-scheme: dark) {
  :root {
    --primary: #818cf8;
    --primary-light: #a5b4fc;
    --bg: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: #1e293b;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --border: #334155;
    --shadow: 0 1px 3px rgba(0,0,0,0.3);
    --shadow-lg: 0 10px 40px rgba(0,0,0,0.4);
  }
}

* { box-sizing: border-box; margin: 0; padding: 0; }

body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg-secondary);
  color: var(--text);
  line-height: 1.6;
  min-height: 100vh;
}

.container {
  max-width: 1000px;
  margin: 0 auto;
  padding: 3rem 2rem;
}

.header {
  text-align: center;
  margin-bottom: 3rem;
}

.header-icon {
  font-size: 3rem;
  margin-bottom: 1rem;
}

.header h1 {
  font-size: 2.5rem;
  font-weight: 800;
  background: linear-gradient(135deg, var(--primary), var(--primary-light));
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 0.5rem;
}

.header p {
  color: var(--text-muted);
  font-size: 1.1rem;
}

.summary {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 1.5rem;
  margin-bottom: 3rem;
}

.summary-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  text-align: center;
  box-shadow: var(--shadow);
  transition: all 0.3s ease;
}

.summary-card:hover {
  transform: translateY(-4px);
  box-shadow: var(--shadow-lg);
}

.summary-icon {
  font-size: 2rem;
  margin-bottom: 0.5rem;
}

.summary-value {
  font-size: 2.5rem;
  font-weight: 800;
  color: var(--primary);
  line-height: 1.2;
}

.summary-label {
  color: var(--text-muted);
  font-size: 0.9rem;
  margin-top: 0.25rem;
}

.papers-grid {
  display: grid;
  gap: 1rem;
}

.paper-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.5rem;
  cursor: pointer;
  box-shadow: var(--shadow);
  transition: all 0.3s ease;
  overflow: hidden;
}

.paper-card:hover {
  border-color: var(--primary);
  box-shadow: var(--shadow-lg);
}

.paper-card.expanded {
  border-color: var(--primary);
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
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 0.5rem;
  line-height: 1.4;
}

.paper-meta {
  display: flex;
  gap: 1rem;
  font-size: 0.85rem;
  color: var(--text-muted);
  flex-wrap: wrap;
}

.paper-citations {
  display: flex;
  align-items: center;
  gap: 0.75rem;
  flex-shrink: 0;
}

.citation-badge {
  background: linear-gradient(135deg, var(--primary), var(--primary-light));
  color: white;
  padding: 0.5rem 1rem;
  border-radius: 20px;
  font-weight: 700;
  font-size: 1rem;
}

.expand-arrow {
  font-size: 1.25rem;
  color: var(--text-muted);
  transition: transform 0.3s ease;
  flex-shrink: 0;
}

.paper-card.expanded .expand-arrow {
  transform: rotate(180deg);
}

.paper-chart-area {
  max-height: 0;
  overflow: hidden;
  transition: max-height 0.4s ease, margin 0.3s ease;
  margin-top: 0;
}

.paper-card.expanded .paper-chart-area {
  max-height: 400px;
  margin-top: 1.5rem;
}

.chart-container {
  background: var(--bg-secondary);
  border-radius: var(--radius-sm);
  padding: 1.5rem;
  height: 280px;
}

.loading {
  text-align: center;
  padding: 4rem 2rem;
  color: var(--text-muted);
}

.loading-spinner {
  width: 50px;
  height: 50px;
  border: 4px solid var(--border);
  border-top-color: var(--primary);
  border-radius: 50%;
  animation: spin 1s linear infinite;
  margin: 0 auto 1rem;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

@media (max-width: 768px) {
  .container { padding: 2rem 1rem; }
  .header h1 { font-size: 1.8rem; }
  .summary { grid-template-columns: 1fr; gap: 1rem; }
  .summary-card { padding: 1rem; }
  .summary-value { font-size: 2rem; }
  .paper-card { padding: 1rem; }
  .paper-header { flex-direction: column; }
  .paper-citations { width: 100%; justify-content: space-between; }
}
</style>

<div class="container">
  <header class="header">
    <div class="header-icon">📊</div>
    <h1>Citation Analytics</h1>
    <p>Track daily citation changes for each paper</p>
  </header>

  <div class="summary" id="summary">
    <div class="summary-card">
      <div class="summary-icon">📈</div>
      <div class="summary-value" id="totalCitations">-</div>
      <div class="summary-label">Total Citations</div>
    </div>
    <div class="summary-card">
      <div class="summary-icon">📄</div>
      <div class="summary-value" id="totalPapers">-</div>
      <div class="summary-label">Papers</div>
    </div>
    <div class="summary-card">
      <div class="summary-icon">📅</div>
      <div class="summary-value" id="daysTracked">-</div>
      <div class="summary-label">Days Tracked</div>
    </div>
  </div>

  <div class="papers-grid" id="papersGrid">
    <div class="loading">
      <div class="loading-spinner"></div>
      <div>Loading papers...</div>
    </div>
  </div>
</div>

<script>
const CHARTS = {};
let allPapers = [];
let history = {};

async function loadData() {
  try {
    const [dataResp, histResp] = await Promise.all([
      fetch('/citation-tracker/data.json'),
      fetch('/citation-tracker/papers_history.json')
    ]);
    
    if (!dataResp.ok) throw new Error('Failed to load papers');
    if (!histResp.ok) throw new Error('Failed to load history');
    
    const data = await dataResp.json();
    history = await histResp.json();
    allPapers = data.papers || [];
    
    renderSummary(data, history);
    renderPapers();
  } catch (err) {
    console.error(err);
    document.getElementById('papersGrid').innerHTML = `
      <div class="loading">
        <div>⚠️ Failed to load data</div>
        <small style="color: var(--text-muted)">${err.message}</small>
      </div>
    `;
  }
}

function renderSummary(data, history) {
  const total = allPapers.reduce((sum, p) => sum + (p.citations || 0), 0);
  document.getElementById('totalCitations').textContent = (data.total || total).toLocaleString();
  document.getElementById('totalPapers').textContent = allPapers.length;
  document.getElementById('daysTracked').textContent = Object.keys(history).length || 1;
}

function renderPapers() {
  const grid = document.getElementById('papersGrid');
  allPapers.sort((a, b) => (b.citations || 0) - (a.citations || 0));
  
  grid.innerHTML = allPapers.map((paper, idx) => `
    <div class="paper-card" data-idx="${idx}" onclick="togglePaper(${idx})">
      <div class="paper-header">
        <div class="paper-info">
          <div class="paper-title">${escapeHtml(paper.title)}</div>
          <div class="paper-meta">
            <span>${paper.year || 'N/A'}</span>
            ${paper.venue ? `<span>${escapeHtml(paper.venue)}</span>` : ''}
          </div>
        </div>
        <div class="paper-citations">
          <span class="citation-badge">${paper.citations || 0}</span>
          <span class="expand-arrow">▼</span>
        </div>
      </div>
      <div class="paper-chart-area">
        <div class="chart-container">
          <canvas id="chart-${idx}"></canvas>
        </div>
      </div>
    </div>
  `).join('');
}

function togglePaper(idx) {
  const cards = document.querySelectorAll('.paper-card');
  const card = document.querySelector(`[data-idx="${idx}"]`);
  const isExpanded = card.classList.contains('expanded');
  
  // Close others
  cards.forEach(c => {
    if (c !== card && c.classList.contains('expanded')) {
      c.classList.remove('expanded');
      const i = parseInt(c.dataset.idx);
      if (CHARTS[i]) {
        CHARTS[i].destroy();
        delete CHARTS[i];
      }
    }
  });
  
  // Toggle current
  card.classList.toggle('expanded');
  
  if (card.classList.contains('expanded') && !CHARTS[idx]) {
    setTimeout(() => renderChart(idx), 100);
  }
}

function renderChart(idx) {
  const paper = allPapers[idx];
  const pid = paper.id || paper.title;
  const dates = Object.keys(history).sort();
  
  const data = dates.map(d => {
    const day = history[d]?.[pid];
    return day?.citations ?? null;
  });
  
  const ctx = document.getElementById(`chart-${idx}`);
  if (!ctx) return;
  
  if (CHARTS[idx]) CHARTS[idx].destroy();
  
  CHARTS[idx] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        data: data,
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        fill: true,
        tension: 0.4,
        pointRadius: 5,
        pointHoverRadius: 7,
        pointBackgroundColor: '#6366f1',
        pointBorderColor: '#fff',
        pointBorderWidth: 2
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: { grid: { display: false } },
        y: { beginAtZero: false, grid: { color: 'rgba(0,0,0,0.05)' } }
      }
    }
  });
}

function escapeHtml(text) {
  const d = document.createElement('div');
  d.textContent = text;
  return d.innerHTML;
}

loadData();
</script>
