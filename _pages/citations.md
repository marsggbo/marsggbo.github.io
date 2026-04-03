---
layout: page
title: Citation Analytics
permalink: /citations/
nav: true
nav_order: 99
---

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>

<style>
.analytics-container { max-width: 900px; margin: 0 auto; padding-top: 1rem; }
.stat-summary { display: flex; gap: 1.5rem; margin-bottom: 2rem; background: var(--global-bg-color, white); border-radius: 12px; padding: 1.5rem; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border: 1px solid var(--global-divider-color, #e2e8f0); }
.stat-box { flex: 1; text-align: center; }
.stat-val { font-size: 2rem; font-weight: bold; color: #667eea; }
.stat-label { font-size: 0.9rem; color: var(--global-text-color-light, #64748b); text-transform: uppercase; letter-spacing: 1px;}
.paper-card { background: var(--global-bg-color, white); border: 1px solid var(--global-divider-color, #e2e8f0); border-radius: 12px; padding: 1.5rem; margin-bottom: 1rem; cursor: pointer; transition: all 0.2s; }
.paper-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }
.paper-header { display: flex; justify-content: space-between; align-items: flex-start; gap: 1rem; }
.paper-title { font-weight: 600; font-size: 1.1rem; flex: 1; line-height: 1.4;}
.paper-cites { color: #667eea; font-weight: bold; font-size: 1.25rem; display: flex; align-items: center; gap: 0.5rem; }
.paper-cites span { font-size: 0.8rem; background: rgba(102, 126, 234, 0.1); padding: 0.2rem 0.6rem; border-radius: 999px; }
.chart-wrapper { display: none; padding-top: 1.5rem; margin-top: 1rem; border-top: 1px solid var(--global-divider-color, #e2e8f0); height: 250px; }
.expanded .chart-wrapper { display: block; }
.toggle-icon { font-size: 1.25rem; opacity: 0.5; transition: transform 0.3s; }
.expanded .toggle-icon { transform: rotate(180deg); }
#loading { text-align: center; padding: 3rem; color: #64748b; font-size: 1.2rem; }
</style>

<div class="analytics-container">
  <h1>📊 Citation Analytics</h1>
  <p style="color: var(--global-text-color-light, #64748b); margin-bottom: 2rem;">Tracks daily citation counts from Google Scholar to visualize impact over time.</p>
  
  <div id="loading">Gathering citation intelligence...</div>
  
  <div id="dashboard" style="display: none;">
    <div class="stat-summary">
        <div class="stat-box">
            <div class="stat-val" id="totalCites">0</div>
            <div class="stat-label">Total Citations</div>
        </div>
        <div class="stat-box">
            <div class="stat-val" id="totalPapers">0</div>
            <div class="stat-label">Tracked Papers</div>
        </div>
        <div class="stat-box">
            <div class="stat-val" id="lastUpdated">-</div>
            <div class="stat-label">Last Updated</div>
        </div>
    </div>
    
    <div id="papersList"></div>
  </div>
</div>

<script>
let globalDates = [];
let allPapers = [];
let charts = {};

fetch('/citations.json')
  .then(res => res.json())
  .then(data => {
    let datesSet = new Set();
    
    for (let title in data) {
        let cites = data[title].citations || {};
        for (let d in cites) {
            datesSet.add(d);
        }
    }
    globalDates = Array.from(datesSet).sort();
    
    let sum = 0;
    for (let title in data) {
        let citesDict = data[title].citations || {};
        let dataArray = [];
        let latestCite = 0;
        
        for (let d of globalDates) {
            if (citesDict[d] !== undefined) {
                latestCite = citesDict[d];
            }
            dataArray.push(latestCite);
        }
        
        sum += latestCite;
        
        allPapers.push({
            title: title,
            citations: dataArray,
            total_citations: latestCite
        });
    }
    
    allPapers.sort((a, b) => b.total_citations - a.total_citations);
    
    document.getElementById('totalCites').innerText = sum;
    document.getElementById('totalPapers').innerText = allPapers.length;
    document.getElementById('lastUpdated').innerText = globalDates.length > 0 ? globalDates[globalDates.length - 1] : 'N/A';
    
    renderPapers();
    
    document.getElementById('loading').style.display = 'none';
    document.getElementById('dashboard').style.display = 'block';
  })
  .catch(err => {
    document.getElementById('loading').innerHTML = `⚠️ Failed to load citation data. Please check if the GitHub Action ran successfully.`;
    console.error(err);
  });

function renderPapers() {
  const container = document.getElementById('papersList');
  let html = '';
  
  allPapers.forEach((paper, index) => {
    html += `
      <div class="paper-card" onclick="togglePaper(${index})">
        <div class="paper-header">
            <div class="paper-title">${paper.title}</div>
            <div class="paper-cites">
                ${paper.total_citations} <span>cites</span>
                <div class="toggle-icon">▼</div>
            </div>
        </div>
        <div class="chart-wrapper">
            <canvas id="chart-${index}"></canvas>
        </div>
      </div>
    `;
  });
  
  container.innerHTML = html;
}

function togglePaper(index) {
  const cards = document.querySelectorAll('.paper-card');
  const card = cards[index];
  
  card.classList.toggle('expanded');
  
  if (card.classList.contains('expanded') && !charts[index]) {
    // Small delay to ensure the canvas is visible so Chart.js computes height properly
    setTimeout(() => {
        const ctx = document.getElementById(`chart-${index}`).getContext('2d');
        const paper = allPapers[index];
        
        charts[index] = new Chart(ctx, {
            type: 'line',
            data: {
                labels: globalDates,
                datasets: [{
                    label: 'Citations',
                    data: paper.citations,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    borderWidth: 3,
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
                    legend: { display: false },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    }
                },
                scales: {
                    y: {
                        ticks: { stepSize: 1, precision: 0 },
                        suggestedMin: Math.max(0, paper.total_citations - 5),
                        suggestedMax: paper.total_citations + 5
                    }
                }
            }
        });
    }, 50);
  }
}
</script>
