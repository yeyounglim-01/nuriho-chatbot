import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import ChatMessage

# ------------------- Azure 설정 -------------------
azure_endpoint = st.secrets["AZURE_OAI_ENDPOINT"].rstrip("/")  # 혹시라도 끝에 / 있으면 제거
azure_key = st.secrets["AZURE_OAI_KEY"]
llm_deployment = st.secrets["AZURE_OAI_DEPLOYMENT"]

# LLM (gpt-4o-mini)
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_key,
    azure_deployment=llm_deployment,
    api_version="2024-05-01-preview",   # 이 버전이 제일 안정적
    temperature=0.3,
    max_tokens=1000
)

# Embedding (여기서 이름만 정확히 맞추면 끝!)
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=azure_endpoint,
    api_key=azure_key,
    azure_deployment="ada",             # 너가 방금 만든 이름이 ada면 이걸로!
    # 만약 이름이 다르면 여기만 바꿔 → 예: "text-embedding-ada-002", "my-ada" 등
    api_version="2024-05-01-preview",   # 이 버전이 embedding에서 제일 잘 됨
)

# ------------------- UI -------------------
st.set_page_config(page_title="누리호 백과사전", page_icon="🚀")
st.title("누리호(KSLV-II) 백과사전 챗봇")

if "messages" not in st.session_state:
    st.session_state.messages = [
        ChatMessage(role="assistant", content="""
안녕하세요! 저는 **누리호 1차부터 4차 발사, 탑재위성 교신 결과까지 전부 알고 있는 전문 챗봇**입니다!  
자유롭게 물어보시거나 아래 주제를 골라주세요!""")
    ]

cols = st.columns(3)
menus = [
    ("누리호 뜻과 목표", "누리호 이름의 뜻과 개발 목표 알려줘"),
    ("1차 발사", "1차 발사 때 무슨 일 있었어?"),
    ("2차 발사", "2차 발사 과정 설명해줘"),
    ("3차 발사 성공", "3차 발사 성공했지? 과정이 어땠어?"),
    ("4차 발사 성공", "4차 발사는 언제 했고 성공했어?"),
    ("4차 위성 교신", "4차 때 올린 위성들 교신 잘 돼?")
]
for i, (label, q) in enumerate(menus):
    with cols[i % 3]:
        if st.button(label, use_container_width=True):
            st.session_state.messages.append(ChatMessage(role="user", content=q))
            st.rerun()

# ------------------- 벡터DB (핵심 수정: 강제로 재생성 방지 + 로딩 메시지) -------------------
@st.cache_resource(show_spinner="누리호 자료를 열심히 읽고 있어요... 잠시만 기다려주세요 🚀")
def get_retriever():
    with st.spinner("PDF를 읽고 벡터DB 만드는 중... (최초 1회만 걸려요!)"):
        loader = PyPDFDirectoryLoader("data/")
        docs = loader.load()
        if not docs:
            st.error("data 폴더에 PDF 파일이 없어요! 확인해주세요!")
            st.stop()

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
    ("system", """너는 대한민국 누리호 전문가야. 주어진 문서만 보고 정확하고 따뜻하게 한국어로 답해.
관련 문서: {context}"""),
    MessagesPlaceholder("history"),
    ("human", "{question}")
])

chain = (
    {"context": retriever, "question": RunnablePassthrough(), "history": lambda x: st.session_state.messages[-10:]}
    | prompt
    | llm
    | StrOutputParser()
)

# ------------------- 채팅 -------------------
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if user_input := st.chat_input("누리호에 대해 궁금한 거 다 물어보세요!"):
    st.session_state.messages.append(ChatMessage(role="user", content=user_input))
    st.chat_message("user").write(user_input)

    with st.chat_message("assistant"):
        response = chain.stream(user_input)
        answer = st.write_stream(response)
    st.session_state.messages.append(ChatMessage(role="assistant", content=answer))
