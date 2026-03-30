---
layout: page
title: Citation Analytics
permalink: /citations/
nav: true
nav_order: 99
---

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
:root { --primary: #3b82f6; --bg: #f8fafc; --text: #1e293b; --card: #fff; }
@media (prefers-color-scheme: dark) { :root { --bg: #0f172a; --text: #f1f5f9; --card: #1e293b; } }
body { background: var(--bg); color: var(--text); font-family: system-ui; padding: 2rem; max-width: 1200px; margin: 0 auto; }
a { color: var(--primary); }
.stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1rem; margin: 2rem 0; }
.stat { background: var(--card); padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.stat h3 { margin: 0; font-size: 2rem; color: var(--primary); }
.stat p { margin: 0.5rem 0 0; opacity: 0.7; }
.chart { background: var(--card); padding: 1.5rem; border-radius: 12px; margin: 1rem 0; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
th, td { padding: 0.75rem; text-align: left; border-bottom: 1px solid #e2e8f0; }
th { background: var(--card); font-weight: 600; }
.cites { color: var(--primary); font-weight: bold; }
.loading { text-align: center; padding: 2rem; opacity: 0.5; }
</style>

<div class="stats">
  <div class="stat"><h3 id="total">-</h3><p>Total Citations</p></div>
  <div class="stat"><h3 id="hindex">-</h3><p>h-index</p></div>
  <div class="stat"><h3 id="i10">-</h3><p>i10-index</p></div>
  <div class="stat"><h3 id="papers">-</h3><p>Papers</p></div>
</div>

<div class="chart"><canvas id="trendChart" height="80"></canvas></div>
<div class="chart"><canvas id="barChart" height="150"></canvas></div>

<h3>📚 All Papers</h3>
<table><thead><tr><th>#</th><th>Title</th><th>Year</th><th>Citations</th></tr></thead>
<tbody id="papersTable"><tr><td colspan="4" class="loading">Loading...</td></tr></tbody></table>

<script>
// 加载数据
Promise.all([
  fetch('/assets/scholar-tracker/data.json').then(r=>r.json()).catch(()=>null),
  fetch('/assets/scholar-tracker/history.json').then(r=>r.json()).catch(()=>{history:{}})
]).then(([data, historyData]) => {
  if (!data) return;
  
  // 更新统计
  document.getElementById('total').textContent = data.total.toLocaleString();
  document.getElementById('hindex').textContent = data.hindex;
  document.getElementById('i10').textContent = data.i10index;
  document.getElementById('papers').textContent = data.papers.length;
  
  // 渲染表格
  const tbody = document.getElementById('papersTable');
  tbody.innerHTML = data.papers.map((p,i)=>`<tr><td>${i+1}</td><td>${p.title}</td><td>${p.year}</td><td class="cites">${p.citations}</td></tr>`).join('');
  
  // 趋势图
  const hist = historyData?.history || [];
  if (hist.length > 0) {
    new Chart(document.getElementById('trendChart'), {
      type: 'line',
      data: { labels: hist.map(h=>h.date), datasets: [{ label:'Citations', data: hist.map(h=>h.total), borderColor:'#3b82f6', fill:false, tension:0.3 }] },
      options: { responsive: true, plugins: { legend: {display:false} }, scales: { x: {grid:{display:false}}, y: {grid:{display:false}} } } }
    });
  }
  
  // 柱状图
  const top10 = data.papers.slice(0,10);
  new Chart(document.getElementById('barChart'), {
    type: 'bar',
    data: { labels: top10.map(p=>p.title.substring(0,30)+'...'), datasets: [{ label:'Citations', data: top10.map(p=>p.citations), backgroundColor:'#3b82f6' }] },
    options: { indexAxis:'y', responsive: true, plugins: { legend: {display:false} } }
  });
});
</script>
