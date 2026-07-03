# 🧠 AI 大脑可视化仪表盘（AI-Monitor）

> 像心电图仪一样监控 AI 的每一次心跳 —— 记录行为、画出轨迹、揪出幻觉。

## 项目简介

AI-Monitor 是一个轻量级的 AI 行为可视化仪表盘，通过记录每次 API 调用，用图表展示 AI 的注意力分布、记忆衰减和逻辑矛盾，帮你一眼看穿 AI 有没有"精神分裂"。

## 核心功能

### 7 大面板

| 面板 | 说明 |
|------|------|
| 💬 **对话** | GPT 风格聊天窗口，支持 `/tag` 标签分类，自动记录完整对话 |
| 🌟 **记忆星空图** | 关键词关联网络图，展示 AI 对不同概念的关注和连接 |
| 🔥 **注意力热力图** | 词级别关注度可视化，深色=高关注，一眼看出 AI 忽略了什么 |
| 📉 **记忆衰减曲线** | 随对话推移的记忆准确度趋势，曲线下跌 = 幻觉风险上升 |
| 🧠 **模型诊断** | Token 消耗、响应延迟等运行指标监控 |
| 🛡 **模型验证** | 回答质量评估与一致性检查 |
| 🚨 **预警中心** | 自动检测逻辑矛盾和历史引用错误，标红提醒 |

### 多供应商支持

- **OpenAI** — 支持 logprobs，注意力热力图数据源
- **DeepSeek** — 性价比高，国内直连
- **Qwen（通义千问）** — 阿里云，国内直连
- 兼容任何 OpenAI 接口的服务

## 技术栈

| 模块 | 技术 |
|------|------|
| 可视化框架 | Streamlit 1.37 |
| 图表库 | Plotly 5.23 |
| 数据存储 | SQLite |
| NLP 分词 | jieba |
| API 调用 | OpenAI SDK |
| 数据分析 | NumPy, scikit-learn |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 配置 API

编辑 `config.yaml`，填入你的 API Key：

```yaml
providers:
  - name: "DeepSeek"
    base_url: "https://api.deepseek.com/v1"
    api_key: "sk-你的key"
    model: "deepseek-chat"

active_provider: 0   # 0 表示使用第一个供应商
```

### 3. 启动

```bash
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

### 4. 使用

1. 在侧边栏选择 API 供应商，点击"测试连接"
2. 切换到「对话」标签，像平时一样聊天
3. 用 `/tag 项目名` 给对话打标签分类
4. 切换到其他标签页查看可视化图表

## 项目结构

```
ai_Brain_CT/
├── app.py                  # Streamlit 主入口，多标签页路由
├── config.yaml             # API 配置文件（含 Key，已 gitignore）
├── requirements.txt        # Python 依赖
├── README.md               # 项目文档
├── src/
│   ├── analyzer.py         # 分析引擎：关键词提取、矛盾检测
│   ├── api_client.py       # 多供应商 API 客户端工厂
│   ├── config.py           # 配置加载与解析
│   ├── database.py         # SQLite 数据库操作
│   └── panels/
│       ├── chat_panel.py       # 对话面板
│       ├── star_map.py         # 星空图面板
│       ├── heatmap.py          # 热力图面板
│       ├── decay_curve.py      # 衰减曲线面板
│       ├── model_diagnostics.py # 模型诊断面板
│       ├── model_verification.py # 模型验证面板
│       └── alerts_panel.py     # 预警中心面板
└── utils/
    └── helpers.py          # 工具函数
```

## 版本说明

- **Python**: 3.10+
- **Streamlit**: 锁定 1.37.0（1.58+ 与 Plotly 存在 JS 兼容性问题）
- **Plotly**: 锁定 5.23.0（6.x 与 Streamlit 1.37 不兼容）

## License

MIT
