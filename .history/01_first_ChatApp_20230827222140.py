import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage,  # 人間の質問
    AIMessage  # ChatGPTの返答
)

def main():
    llm = ChatOpenAI(temperature=0)  # ChatGPT APIを呼んでくれる機能

    # ウェブページの設定
    st.set_page_config(
        page_title="My First ChatGPT Apps",
        page_icon="🤗"
    )
    st.header("My First ChatGPT Apps🤗")

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="絶対に関西弁で返答してください.")
        ]

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        with st.spinner("ChatGPT is typing ..."):
            response = llm(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

    # チャット履歴の表示
    messages = st.session_state.get('messages', [])
    for message in messages:
        if isinstance(message, AIMessage):
            with st.chat_message('assistant'):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message('user'):
                st.markdown(message.content)
        else:  # isinstance(message, SystemMessage):
            st.write(f"System message: {message.content}")


if __name__ == '__main__':
    main()


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
    SystemMessage(content="絶対に関西弁で返答してください"),
    HumanMessage(content=message)
]
response = llm(messages)
print(response)