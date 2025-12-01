import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage
import os

# ------------------- Azure 설정 (secrets.toml에서 자동 로드) -------------------
azure_endpoint = st.secrets["AZURE_OAI_ENDPOINT"]
azure_key = st.secrets["AZURE_OAI_KEY"]
azure_deployment = st.secrets["AZURE_OAI_DEPLOYMENT"]

# LLM & 임베딩
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_key,
    azure_deployment=azure_deployment,
    api_version="2024-08-01-preview",
    temperature=0.3,
    max_tokens=1000
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    api_key=azure_key,
    azure_deployment=st.secrets.get("AZURE_EMBEDDING_DEPLOYMENT", "ada"),
    api_version="2024-08-01-preview"
)

# ------------------- 페이지 설정 -------------------
st.set_page_config(page_title="누리호 백과사전", page_icon="🚀")
st.title("🚀 누리호(KSLV-II) 공식급 백과사전 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = [
        ChatMessage(role="assistant", content="""
안녕하세요! 저는 **누리호 1차 발사부터 2025년 4차 완벽 성공, 탑재위성 교신 결과까지 전부 알고 있는 대한민국 대표 우주 챗봇**입니다!  

자유롭게 질문하거나 아래 주제를 골라주세요!
        """)
    ]

# 예쁜 메뉴 버튼들
cols = st.columns(3)
menus = [
    ("누리호의 뜻과 목표", "누리호 이름 뜻과 개발 목표 알려줘"),
    ("1차 발사 (2021.10.21)", "1차 발사 때 무슨 일 있었어?"),
    ("2차 발사 (2022.6.21)", "2차 발사 과정 설명해줘"),
    ("3차 발사 성공 (2023.5.25)", "3차 발사는 성공했지? 과정이 어땠어?"),
    ("4차 발사 성공 (2025)", "최근 4차 발사는 언제 했고, 왜 그 날짜였어? 성공했어?"),
    ("4차 탑재위성 교신 결과", "4차 때 쏜 위성들 지금 교신 잘 돼?")
]

for i, (label, q) in enumerate(menus):
    with cols[i % 3]:
        if st.button(label, use_container_width=True):
            st.session_state.messages.append(ChatMessage(role="user", content=q))
            st.rerun()

# ------------------- 벡터DB 로드 (최초 1회만 생성) -------------------
@st.cache_resource
def get_retriever():
    loader = PyPDFDirectoryLoader("data/")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory="./vectorstore"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 6})

retriever = get_retriever()

# ------------------- RAG 체인 -------------------
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    당신은 대한민국 누리호(KSLV-II) 최고 전문가입니다.
    아래 문서들만을 기반으로 정확하고 친절하게 답변하세요.
    1~4차 발사, 발사 시각 선정 이유, 탑재위성 교신 결과까지 전부 정확히 알고 있습니다.
    답변은 자연스럽고 따뜻한 한국어로 해주세요.

    관련 문서:
    {context}
    """),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

chain = (
    {"context": retriever, "question": RunnablePassthrough(), "history": lambda x: st.session_state.messages[-10:]}
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------- 채팅 표시 & 입력 -------------------
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input("누리호에 대해 궁금한 거 다 물어보세요! 🚀"):
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    with st.chat_message("assistant"):
        response = chain.stream(prompt)
        answer = st.write_stream(response)
    

    st.session_state.messages.append(ChatMessage(role="assistant", content=answer))
