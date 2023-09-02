import streamlit as st

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.callbacks import StreamlitCallbackHandler

from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage,  # 人間の質問
    AIMessage  # ChatGPTの返答
)
from langchain.chat_models import ChatOpenAI


def init_page():
    # ウェブページの設定
    st.set_page_config(
        page_title="複数アプリ連携ページ",
        page_icon="🤗"
    )
    st.header("複数アプリ連携ページ🤗")
    # st.sidebar.title("Options")


def main():
    init_page()


if __name__ == '__main__':
    main()
