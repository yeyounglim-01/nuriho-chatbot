import streamlit as st
from openai import AzureOpenAI
import os
import time

# ------------------- Azure Assistants 설정 -------------------
client = AzureOpenAI(
    azure_endpoint=st.secrets["AZURE_OPENAI_ENDPOINT"],
    api_key=st.secrets["AZURE_OPENAI_API_KEY"],
    api_version="2024-05-01-preview"
)

# 너가 만든 Assistant ID랑 Vector Store ID (이거만 바꾸면 됨!)
ASSISTANT_ID = "asst_l34cDmbOAWtDk2FOQjNQ0WMD"  # ← 너가 Azure AI Studio에서 만든 assistant의 ID
VECTOR_STORE_ID = "vs_aNbtteNGCu6QJEHKHYs6J7Bg"  # 너가 이미 써놓은 거 그대로!

# ------------------- Streamlit UI -------------------
st.set_page_config(page_title="누리호 백과사전", page_icon="로켓")
st.title("로켓 누리호(KSLV-II) 완전 정복 챗봇")
st.markdown("**1차 발사부터 4차 성공, 탑재위성 교신까지 전부 알고 있어요!**")

# 세션에 thread 없으면 만들기
if "thread_id" not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id
    st.session_state.messages = []

# 메뉴 버튼들
cols = st.columns(3)
menus = [
    ("누리호 뜻", "누리호 이름의 뜻이 뭐야?"),
    ("1차 발사", "1차 발사 때 무슨 일 있었어?"),
    ("4차 발사", "4차 발사는 언제 했고 성공했어?"),
    ("위성 교신", "4차에 실은 위성들 지금 교신 잘 돼?"),
    ("개발 목표", "누리호 개발 목표가 뭐였어?"),
    ("다음 발사", "누리호 다음 발사는 언제야?")
]
for i, (label, q) in enumerate(menus):
    with cols[i % 3]:
        if st.button(label, use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

# 채팅 기록 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 사용자 입력
if prompt := st.chat_input("누리호에 대해 궁금한 거 다 물어보세요!"):
    # 사용자 메시지 추가
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Assistant에게 전달
    client.beta.threads.messages.create(
        thread_id=st.session_state.thread_id,
        role="user",
        content=prompt
    )

    # Run 실행
    with st.chat_message("assistant"):
        with st.spinner("누리호 전문가가 열심히 답변 작성 중..."):
            run = client.beta.threads.runs.create(
                thread_id=st.session_state.thread_id,
                assistant_id=ASSISTANT_ID
            )

            # 완료될 때까지 대기
            while run.status in ["queued", "in_progress"]:
                time.sleep(0.5)
                run = client.beta.threads.runs.retrieve(thread_id=st.session_state.thread_id, run_id=run.id)

            # 답변 가져오기
            messages = client.beta.threads.messages.list(thread_id=st.session_state.thread_id)
            assistant_reply = messages.data[0].content[0].text.value

            st.write(assistant_reply)
            st.session_state.messages.append({"role": "assistant", "content": assistant_reply})

# 하단에 재미로
st.markdown("---")
st.caption("Made with 로켓 by 대한민국 우주 덕후")
