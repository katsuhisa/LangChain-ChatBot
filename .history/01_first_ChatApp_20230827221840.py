import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage,  # 人間の質問
    AIMessage  # ChatGPTの返答
)

# ウェブページの設定
st.set_page_config(
    page_title="My First ChatGPT Apps",
    page_icon="🤗"
)
st.header("My First ChatGPT Apps🤗")

if user_input := st.chat_input("Ask me anything!"):
    # なにか入力されればここが実行される

container = st.container()
with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area(label='Message: ', key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        # 何か入力されて Submit ボタンが押されたら実行される

with st.spinner("ChatGPT is typing ..."):
    response = llm(st.session_state.messages)


llm = ChatOpenAI()  # ChatGPT APIを呼んでくれる機能
message = "Hi, ChatGPT!"  # あなたの質問をここに書く

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content=message)
]
response = llm(messages)
print(response)

# content='Hello! How can I assist you today?' additional_kwargs={} example=False