---
layout: page
title: Citation Analytics
permalink: /citations/
nav: true
nav_order: 99
---

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1"></script>

<style>
body { font-family: -apple-system, sans-serif; padding: 2rem; background: #f8fafc; }
@media (prefers-color-scheme: dark) { body { background: #0f172a; color: #f1f5f9; } }
.container { max-width: 800px; margin: 0 auto; }
.card { background: white; border: 1px solid #e2e8f0; border-radius: 12px; padding: 1rem; margin-bottom: 0.5rem; cursor: pointer; }
@media (prefers-color-scheme: dark) { .card { background: #1e293b; border-color: #334155; } }
.paper { display: flex; justify-content: space-between; }
.title { font-weight: 600; }
.cites { color: #6366f1; font-weight: bold; }
.chart { display: none; padding-top: 1rem; }
.expanded .chart { display: block; }
</style>

<div class="container">
  <h1>📊 Citation Analytics</h1>
  <div id="loading">Loading...</div>
  <div id="papers"></div>
</div>

<script>
var papers = [];
var historyData = {};
var charts = {};

Promise.all([
  fetch('/data.json').then(function(r) { return r.json(); }),
  fetch('/papers_history.json').then(function(r) { return r.json(); })
]).then(function(results) {
  papers = results[0].papers || [];
  historyData = results[1] || {};
  
  var total = 0;
  for (var i = 0; i < papers.length; i++) total += papers[i].citations || 0;
  document.getElementById('loading').innerHTML = '<p>Total: ' + total + ' | Papers: ' + papers.length + '</p>';
  
  papers.sort(function(a, b) { return (b.citations || 0) - (a.citations || 0); });
  
  var html = '';
  for (var i = 0; i < papers.length; i++) {
    html += '<div class="card" onclick="toggle(' + i + ')">';
    html += '<div class="paper"><span class="title">' + papers[i].title + '</span><span class="cites">' + (papers[i].citations || 0) + '</span></div>';
    html += '<div class="chart"><canvas id="c' + i + '"></canvas></div>';
    html += '</div>';
  }
  document.getElementById('papers').innerHTML = html;
});

function toggle(i) {
  var card = document.querySelectorAll('.card')[i];
  card.classList.toggle('expanded');
  if (card.classList.contains('expanded') && !charts[i]) {
    setTimeout(function() { drawChart(i); }, 100);
  }
}

function drawChart(i) {
  var p = papers[i];
  var pid = p.title;
  var dates = Object.keys(historyData).sort();
  var data = dates.map(function(d) { return (historyData[d] && historyData[d][pid]) ? historyData[d][pid].citations : 0; });
  
  charts[i] = new Chart(document.getElementById('c' + i), {
    type: 'line',
    data: { labels: dates, datasets: [{ data: data, borderColor: '#6366f1', fill: false }] },
    options: { responsive: true, maintainAspectRatio: false }
  });
}
</script>
