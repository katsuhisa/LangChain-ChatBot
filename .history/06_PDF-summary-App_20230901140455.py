import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter


def init_page():
    # ウェブページの設定
    st.set_page_config(
        page_title="PDF Summarizer Apps",
        page_icon="🚀"
    )
    st.header("PDF Summarizer Apps📄")
    st.subheader("15分以上の動画を要約するときは、サイドバーで:blue[GPT-3.5-16k]を選択してね")
    st.sidebar.title("Options")
    # st.session_state.model = "gpt-3.5-turbo-16k-0613"
    st.session_state.costs = []
    st.session_state.prompt_tokens = []
    st.session_state.answer_tokens = []


def select_model():
    model = st.sidebar.selectbox(
        "今回使用するモデルは:",
        ("GPT-3.5", "GPT-3.5-16k", "GPT-4"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-0613"
    elif model == "GPT-3.5-16k":
        st.session_state.model_name = "gpt-3.5-turbo-16k-0613"
    else:
        st.session_state.model_name = "gpt-4"

    # サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    temperature = st.sidebar.slider(
        "回答の創造性:", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    st.session_state.max_token = OpenAI.modelname_to_contextsize(
        st.session_state.model_name) - 300
    # st.session_state.overlap = st.sidebar.slider(
    #     "チャンク間の重複文字数:", min_value=0, max_value=100, value=0, step=10)
    return ChatOpenAI(temperature=temperature, model_name=st.session_state.model_name)


def get_pdf_text():
    uploaded_file = st.file_uploader(
        label='Upload your PDF here😇',
        type='pdf'  # アップロードを許可する拡張子 (複数設定可)
    )
    if uploaded_file:
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name=st.session_state.emb_model_name,
            # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
            # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
            # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
            chunk_size=250,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None
