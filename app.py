
# 📦 项目名称：AI广告文案生成助手（低代码 LangChain Agent 进阶版）
# 💻 技术栈：LangChain + OpenAI + Streamlit + Python
# ✅ 功能：对话式广告文案生成，支持语气选择、多语言、本地品牌知识接入、记忆功能、多轮交互

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains.retrieval_qa.base import RetrievalQA
import os
import tempfile
import shutil

# 安全方式读取 API 密钥
openai_api_key = os.getenv("OPENAI_API_KEY", st.secrets.get("openai_api_key", ""))
os.environ["OPENAI_API_KEY"] = openai_api_key

# 初始化LLM和记忆
llm = ChatOpenAI(temperature=0.7)
memory = ConversationBufferMemory(memory_key="chat_history")

# Streamlit界面
st.title("🎯 AI广告文案生成助手（进阶版）")
st.markdown("上传品牌资料 + 多轮交互式输入 + 多语言文案输出")

# 上传品牌文档
uploaded_file = st.file_uploader("📄 上传品牌调性文档（txt格式）", type=["txt"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    loader = TextLoader(tmp_path)
    docs = loader.load()

    # 构建向量索引库
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)

    # 基于RAG构建检索增强生成链
    retriever = vectorstore.as_retriever()
    rag_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
else:
    rag_chain = None

# 交互输入框
if 'product' not in st.session_state:
    st.session_state.product = st.text_input("产品名称")
    st.stop()

if 'features' not in st.session_state:
    st.session_state.features = st.text_area("产品卖点")
    st.stop()

if 'audience' not in st.session_state:
    st.session_state.audience = st.text_input("目标人群")
    st.stop()

platform = st.selectbox("投放平台", ["小红书", "TikTok", "Instagram", "微博"])
tone = st.selectbox("语气风格", ["情绪化", "理性说服", "幽默", "权威"])
language = st.selectbox("输出语言", ["中文", "英文", "日文"])

# Prompt 模板
template = '''
你是一位{platform}广告专家，请为以下产品生成符合{tone}风格的文案。

产品：{product}
卖点：{features}
受众：{audience}
语言：{language}

品牌背景（若有）：{brand_context}

请输出80字以内广告文案。
'''

prompt = PromptTemplate(
    input_variables=["platform", "tone", "product", "features", "audience", "language", "brand_context"],
    template=template,
)

chain = LLMChain(llm=llm, prompt=prompt, memory=memory)

# 按钮触发生成
if st.button("✨ 生成广告文案"):
    brand_context = ""
    if rag_chain:
        brand_context = rag_chain.run("请总结该品牌的核心语调与关键词")

    inputs = {
        "platform": platform,
        "tone": tone,
        "product": st.session_state.product,
        "features": st.session_state.features,
        "audience": st.session_state.audience,
        "language": language,
        "brand_context": brand_context
    }

    with st.spinner("生成中..."):
        result = chain.run(inputs)
        st.success("✅ 文案已生成：")
        st.markdown(f"**{result}**")

# 显示聊天历史
with st.expander("🧾 交互历史记录"):
    st.write(memory.buffer)

st.caption("由 LangChain + OpenAI + RAG + Memory 驱动")
