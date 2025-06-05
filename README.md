# AI广告文案生成助手（Streamlit 部署版）

本项目基于 OpenAI + LangChain + Streamlit 构建，支持以下功能：

- 多语言广告文案生成（中/英/日）
- 支持品牌调性文件上传，自动分析
- A/B语气风格生成 + 平台适配建议
- 多轮交互与对话记忆（LangChain Memory）

## 🛠 部署指南

### 本地运行

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 运行应用：

```bash
streamlit run app.py
```

### 在线部署推荐平台

- [Streamlit Cloud](https://streamlit.io/cloud)
- [Hugging Face Spaces](https://huggingface.co/spaces)

将此文件夹上传至 GitHub 后，可一键部署至 Streamlit Cloud。

## 📌 API Key 设置

请将 `app.py` 中的 OPENAI_API_KEY 替换为你自己的 Key，或在部署平台中设为环境变量。

---

💡 适合用作 AI 产品作品集、简历展示、课程项目。
