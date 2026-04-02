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
  --bg: #f8fafc;
  --card: #ffffff;
  --text: #1e293b;
  --text-muted: #64748b;
  --border: #e2e8f0;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --text: #f1f5f9;
    --text-muted: #94a3b8;
    --border: #334155;
  }
}
body { font-family: -apple-system, sans-serif; background: var(--bg); color: var(--text); margin: 0; padding: 2rem; }
.container { max-width: 900px; margin: 0 auto; }
.header { text-align: center; margin-bottom: 2rem; }
.header h1 { font-size: 2rem; margin: 0; }
.header p { color: var(--text-muted); }
.summary { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; margin-bottom: 2rem; }
.card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; text-align: center; }
.card-value { font-size: 2rem; font-weight: bold; color: var(--primary); }
.card-label { color: var(--text-muted); font-size: 0.9rem; }
.paper-card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1rem; margin-bottom: 0.75rem; cursor: pointer; }
.paper-card:hover { border-color: var(--primary); }
.paper-row { display: flex; justify-content: space-between; align-items: center; }
.paper-title { font-weight: 600; flex: 1; }
.paper-meta { color: var(--text-muted); font-size: 0.85rem; }
.paper-cites { font-weight: bold; color: var(--primary); margin-left: 1rem; }
.chart-area { display: none; padding-top: 1rem; margin-top: 1rem; border-top: 1px solid var(--border); }
.paper-card.expanded .chart-area { display: block; }
.chart-box { height: 200px; background: var(--bg); border-radius: 8px; padding: 1rem; }
.loading { text-align: center; padding: 2rem; color: var(--text-muted); }
</style>

<div class="container">
  <div class="header">
    <h1>📊 Citation Analytics</h1>
    <p>Track daily citation changes</p>
  </div>
  <div class="summary">
    <div class="card"><div class="card-value" id="total">-</div><div class="card-label">Total Citations</div></div>
    <div class="card"><div class="card-value" id="papers">-</div><div class="card-label">Papers</div></div>
    <div class="card"><div class="card-value" id="days">-</div><div class="card-label">Days</div></div>
  </div>
  <div id="list"><div class="loading">Loading...</div></div>
</div>

<script>
var papers = [];
var historyData = {};
var charts = {};

Promise.all([
  fetch('/data.json').then(function(r) { return r.json(); }).catch(function() { return null; }),
  fetch('/papers_history.json').then(function(r) { return r.json(); }).catch(function() { return null; })
]).then(function(results) {
  var data = results[0];
  var hist = results[1];
  
  if (!data || !data.papers) {
    document.getElementById('list').innerHTML = '<div class="loading">No data available</div>';
    return;
  }
  
  papers = data.papers;
  historyData = hist || {};
  
  var total = 0;
  for (var i = 0; i < papers.length; i++) {
    total += papers[i].citations || 0;
  }
  
  document.getElementById('total').textContent = total;
  document.getElementById('papers').textContent = papers.length;
  document.getElementById('days').textContent = Object.keys(historyData).length || 1;
  
  // Sort by citations
  papers.sort(function(a, b) { return (b.citations || 0) - (a.citations || 0); });
  
  var list = document.getElementById('list');
  var html = '';
  for (var i = 0; i < papers.length; i++) {
    var p = papers[i];
    html += '<div class="paper-card" onclick="togglePaper(' + i + ')">';
    html += '<div class="paper-row">';
    html += '<div>';
    html += '<div class="paper-title">' + escapeHtml(p.title) + '</div>';
    html += '<div class="paper-meta">' + (p.year || '') + ' ' + (p.venue || '') + '</div>';
    html += '</div>';
    html += '<div class="paper-cites">' + (p.citations || 0) + '</div>';
    html += '</div>';
    html += '<div class="chart-area"><div class="chart-box"><canvas id="chart' + i + '"></canvas></div></div>';
    html += '</div>';
  }
  list.innerHTML = html;
}).catch(function(e) {
  document.getElementById('list').innerHTML = '<div class="loading">Error: ' + e.message + '</div>';
});

function togglePaper(index) {
  var cards = document.querySelectorAll('.paper-card');
  var card = cards[index];
  card.classList.toggle('expanded');
  if (card.classList.contains('expanded') && !charts[index]) {
    setTimeout(function() { drawChart(index); }, 100);
  }
}

function drawChart(index) {
  var p = papers[index];
  var pid = p.id || p.title;
  var dates = Object.keys(historyData).sort();
  var data = [];
  for (var i = 0; i < dates.length; i++) {
    var day = historyData[dates[i]];
    if (day && day[pid]) {
      data.push(day[pid].citations);
    } else {
      data.push(0);
    }
  }
  
  var ctx = document.getElementById('chart' + index);
  if (!ctx) return;
  
  charts[index] = new Chart(ctx, {
    type: 'line',
    data: {
      labels: dates,
      datasets: [{
        data: data,
        borderColor: '#6366f1',
        backgroundColor: 'rgba(99, 102, 241, 0.1)',
        fill: true,
        tension: 0.3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } }
    }
  });
}

function escapeHtml(text) {
  var div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}
</script>
