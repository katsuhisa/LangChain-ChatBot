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
        page_title="生成AIアプリ紹介ページ",
        page_icon="🤗"
    )
    st.header("生成AIアプリ紹介ページ🤗")
    st.subheader("🗣️チャットQAアプリ")

    st.subheader("📄Webサイト要約アプリ")

    st.subheader("🎥Youtube動画要約アプリ")

    st.subheader("📑PDFのQAアプリ")


def main():
    init_page()


if __name__ == '__main__':
    main()
