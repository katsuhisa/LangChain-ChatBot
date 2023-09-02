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
from PIL import Image

image_chat_qa = Image.open('Chat_QA_Apps_architecture.png')
image_web_summary = Image.open('Web-page_summary_Apps_architecture.png')
image_youtube_summary = Image.open('Web-page_summary_Apps_architecture.png')
image_pdf_qa = Image.open('PDF_QA_Apps_architecture.png')


def init_page():
    # ウェブページの設定
    st.set_page_config(
        page_title="生成AIアプリ紹介ページ",
        page_icon="🤗"
    )
    st.header("生成AIアプリ紹介ページ🤗")

    st.subheader("🗣️チャットQAアプリ")
    st.image(image_chat_qa, caption='チャットでQAを行うWebアプリのアーキテクチャー')

    st.subheader("📄Webサイト要約アプリ")
    st.image(image_web_summary, caption='Webページの要約を行うWebアプリのアーキテクチャー')

    st.subheader("🎥Youtube動画要約アプリ")
    st.image(image_youtube_summary, caption='Youtube動画の要約を行うWebアプリのアーキテクチャー')

    st.subheader("📑PDFのQAアプリ")
    st.image(image_pdf_qa, caption='PDFをアップロードしてQAを行うWebアプリのアーキテクチャー')


def main():
    init_page()


if __name__ == '__main__':
    main()
