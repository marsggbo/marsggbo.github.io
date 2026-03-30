# 📊 Citation Tracker

自动追踪 Google Scholar 论文引用量的可视化工具。

## 功能

- 🤖 **每日自动更新** - GitHub Actions 每天早上 10 点自动抓取
- 📈 **引用趋势图** - 展示总引用数随时间的变化
- 📚 **论文排名** - Top 10 高引用论文柱状图
- 📄 **完整列表** - 所有论文的引用数统计
- 🔄 **零成本** - 纯 GitHub Actions 实现，无需服务器

## 设置

### 1. Fork 或使用此模板

### 2. 添加 Secret

在 GitHub 仓库设置中添加：
- `SCHOLAR_USER_ID`: 你的 Google Scholar 用户 ID

例如：`LYNKm_8AAAAJ`

### 3. 启用 GitHub Pages

- Settings → Pages → Source: `main` branch, `/ (root)`

### 4. 查看结果

访问：`https://yourusername.github.io/citation-tracker/`

## 手动触发

在 GitHub Actions 页面点击 "Citation Tracker" → "Run workflow"

## 文件说明

```
├── .github/workflows/
│   └── citation-tracker.yml  # GitHub Actions 工作流
├── scrape_scholar.py         # 爬虫脚本
├── index.html                 # 可视化页面（自动生成）
├── citations_history.csv      # 历史数据（自动生成）
└── README.md
```

## 自定义

修改 `scrape_scholar.py` 中的配置：
- `SCHOLAR_USER_ID`: Google Scholar 用户 ID
- 更新频率: `.github/workflows/citation-tracker.yml` 中的 cron 表达式

## License

MIT
