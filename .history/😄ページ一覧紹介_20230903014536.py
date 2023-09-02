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

image_chat_qa = Image.open('./pages/Chat_QA_Apps_architecture.png')
image_web_summary = Image.open(
    './pages/Web-page_summary_Apps_architecture.png')
image_youtube_summary = Image.open(
    './pages/Web-page_summary_Apps_architecture.png')
image_pdf_qa = Image.open('./pages/PDF_QA_Apps_architecture.png')


def init_page():
    # ウェブページの設定
    st.set_page_config(
        page_title="生成AIアプリ紹介ページ",
        page_icon="🤗"
    )
    st.header("生成AIアプリ紹介ページ🤗")


def apps_category():

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🗣️チャットQA", "📄Webサイト要約", "🎥Youtube動画要約", "📑PDF資料に関するQA"])

    with tab1:
        st.subheader("🗣️チャットQAアプリ")
        st.image(image_chat_qa, caption='チャットでQAを行うWebアプリのアーキテクチャー', width=600)

    with tab2:
        st.subheader("📄Webサイト要約アプリ")
        st.image(image_web_summary,
                 caption='Webページの要約を行うWebアプリのアーキテクチャー', width=600)

    with tab3:
        st.subheader("🎥Youtube動画要約アプリ")
        st.image(image_youtube_summary,
                 caption='Youtube動画の要約を行うWebアプリのアーキテクチャー', width=600)

    with tab4:
        st.subheader("📑PDFのQAアプリ")
        st.image(image_pdf_qa, caption='PDFをアップロードしてQAを行うWebアプリのアーキテクチャー', width=600)


def main():
    init_page()

    st.write("参考記事👇")
    st.write("https://zenn.dev/ml_bear/books/d1f060a3f166a5")
    st.write("\n\n\n\n")
    st.write("ーーーーーーーーーーーーーーーーーーーーーーーーーーーー")

    apps_category()


if __name__ == '__main__':
    main()
