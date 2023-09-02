# OpenAI Embeddings APIでの費用：「PDFのテキストをベクトルDBに保存するとき」「質問を投げるとき」の2か所で発生

# Vector DBをPineconeを使用するように修正
# 複数PDFファイルをアップロードできるように修正

import streamlit as st
from langchain.callbacks import get_openai_callback

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


def init_page():
    # ウェブページの設定
    st.set_page_config(
        page_title="PDF Summarizer Apps",
        page_icon="🚀"
    )
    st.header("PDF Summarizer Apps📄")
    st.subheader("長い資料を要約するときは、サイドバーで:blue[GPT-3.5-16k]を選択してね")
    st.sidebar.title("Navigations")
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


# デフォルトではアップロードされるファイルのサイズは200MBまで. Streamlitのカスタム設定でserver.maxUploadSize設定オプションで変更可能
def get_pdf_text():
    uploaded_file = st.file_uploader(
        label='PDFのアップロードはこちら📁',
        type='pdf',  # アップロードを許可する拡張子 (複数設定可)
        accept_multiple_files='False'  # 複数ファイルのアップロードを許可するかフラグ
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


# ベクトルDBの保存場所指定。ローカルファイルシステム、メモリ、フルマネージドクラウドサービス(Qdrant Cloud)、各種On-premise(自社サーバー, AWS, GCPなど)から選択
QDRANT_PATH = "./local_qdrant"  # ローカル保存

# # qdrant cloud への保存 (次の章で詳しく話します)
# client = QdrantClient(
#     url="https://oreno-qdrant-db.us-east-1-0.aws.cloud.qdrant.io:6333",
#     api_key="api-key-hoge123fuga456"
# )

COLLECTION_NAME = "my_collection"

# ベクトルDBを操作するクライアントを準備・設定


def load_qdrant():
    client = QdrantClient(path=QDRANT_PATH)

    # すべてのコレクション（DBのテーブルみたいなもの）名を取得
    collections = client.get_collections().collections
    collection_names = [collection.name for collection in collections]

    # コレクションが存在しなければ作成
    if COLLECTION_NAME not in collection_names:
        # コレクションが存在しない場合、新しく作成します
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )
        print('collection created')

    # 与えられたテキストやDocumentの内容をembeddingsで与えたモデルを用いてEmbedding化する
    # ベクトルDBのclientを用いて、生成したEmbeddingをベクトルDBのcollection_nameに保管
    return Qdrant(
        client=client,
        collection_name=COLLECTION_NAME,
        embeddings=OpenAIEmbeddings()
    )

# PDFのテキストをEmbeddingにしてベクトルDBに保存


def build_vector_store(pdf_text):
    qdrant = load_qdrant()
    qdrant.from_texts(
        texts=pdf_text,
        embeddings=OpenAIEmbeddings(),
        url=QDRANT_PATH,
        api_key="<qdrant-api-key>",
        collection_name=COLLECTION_NAME
    )


def page_pdf_upload_and_build_vector_db():
    st.title("PDF Upload")
    container = st.container()
    with container:
        # さっき作ったもの
        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("Loading PDF ..."):
                build_vector_store(pdf_text)


page_pdf_upload_and_build_vector_db()
