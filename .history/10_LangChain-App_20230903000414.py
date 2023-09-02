import streamlit as st

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.chains import RetrievalQA

from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage,  # 人間の質問
    AIMessage  # ChatGPTの返答
)

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# ベクトルDBの保存場所指定。ローカルファイルシステム、メモリ、フルマネージドクラウドサービス(Qdrant Cloud)、各種On-premise(自社サーバー, AWS, GCPなど)から選択
QDRANT_PATH = "./local_qdrant"  # ローカル保存
COLLECTION_NAME = "my_collection_QA"


def init_page():
    # ウェブページの設定
    st.set_page_config(
        page_title="生成AIアプリ",
        page_icon="🚀"
    )
    st.header("生成AIアプリ📄")
    st.caption("長い資料・動画を要約するときは、サイドバーで:blue[GPT-3.5-16k]を選択してね")
    st.sidebar.title("Navigations")
    # st.session_state.model = "gpt-3.5-turbo-16k-0613"
    st.session_state.costs = []
    st.session_state.prompt_tokens = []
    st.session_state.answer_tokens = []


def select_model():
    model = st.sidebar.selectbox(
        "今回使用するモデルは:",
        ("GPT-3.5", "GPT-3.5-16k"))
    if model == "GPT-3.5":
        st.session_state.model_name = "gpt-3.5-turbo-0613"
    else:
        st.session_state.model_name = "gpt-3.5-turbo-16k-0613"

    # st.session_state.emb_model_name = "gpt-3.5-turbo-16k-0613"  # emb_model_nameを初期化する

    # サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    temperature = st.sidebar.slider(
        "回答の創造性:", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    st.session_state.max_token = OpenAI.modelname_to_contextsize(
        st.session_state.model_name) - 300
    # st.session_state.overlap = st.sidebar.slider(
    #     "チャンク間の重複文字数:", min_value=0, max_value=100, value=0, step=10)
    return ChatOpenAI(temperature=temperature, model_name=st.session_state.model_name)

# クリアボタンを押したときに、メッセージ履歴初期化と同じ処理を走らせると、履歴を消す


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="絶対に関西弁で返答してください.")
        ]
        st.session_state.costs = []


def display_chat_history():
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


def get_answer(llm, messages):  # 回答結果、コスト・プロンプト消費量の算出を行う関数
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost, cb.prompt_tokens, cb.completion_tokens


def calculate_costs():
    costs = st.session_state.get('costs', [])
    st.sidebar.markdown("## Costs")
    st.sidebar.markdown(f"**Total cost: ${sum(costs):.5f}**")
    for cost in costs:
        st.sidebar.markdown(f"- ${cost:.5f}")


def display_tokens():
    prompt_tokens = st.session_state.get('prompt_tokens', [])
    answer_tokens = st.session_state.get('answer_tokens', [])
    st.sidebar.markdown("## Tokens")
    st.sidebar.markdown(f"Prompt Tokens: {prompt_tokens}")
    st.sidebar.markdown(f"Completion Tokens: {answer_tokens}")


def get_pdf_text():  # PDFからtext抽出
    uploaded_file = st.file_uploader(
        label='PDFのアップロードはこちら📁',
        type='pdf',  # アップロードを許可する拡張子 (複数設定可)
        accept_multiple_files=False  # 複数ファイルのアップロードを許可するかフラグ
    )
    if uploaded_file:
        # デフォルトではアップロードされるファイルのサイズは200MBまで. Streamlitのカスタム設定でserver.maxUploadSize設定オプションで変更可能
        pdf_reader = PdfReader(uploaded_file)
        text = '\n\n'.join([page.extract_text() for page in pdf_reader.pages])
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            model_name="text-embedding-ada-002",
            # 適切な chunk size は質問対象のPDFによって変わるため調整が必要
            # 大きくしすぎると質問回答時に色々な箇所の情報を参照することができない
            # 逆に小さすぎると一つのchunkに十分なサイズの文脈が入らない
            chunk_size=250,
            chunk_overlap=0,
        )
        return text_splitter.split_text(text)
    else:
        return None


def load_qdrant():  # ベクトルDBを操作するクライアントを準備・設定
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


def build_vector_store(pdf_text):  # PDFのテキストをEmbeddingにしてベクトルDBに保存

    qdrant = load_qdrant()
    qdrant.add_texts(pdf_text)
    # qdrant.client.close()  # Qdrantクライアントのインスタンスを終了させる


def pdf_upload_and_build_vector_db():  # PDFのアップロード〜ベクトりDB構築
    st.title("PDF(s)独自QAサイト")
    llm = select_model()
    container = st.container()
    with container:

        pdf_text = get_pdf_text()
        if pdf_text:
            with st.spinner("PDFをアップロード中 ..."):
                build_vector_store(pdf_text)
            st.write("アップロードが完了しました。")
            st.write("PDFの内容について質問してください。")


def build_qa_model(llm):
    qdrant = load_qdrant()
    retriever = qdrant.as_retriever(
        search_type="similarity",  # "mmr",  "similarity_score_threshold" などもある
        search_kwargs={"k": 10}  # 文書を何個取得するか (default: 4)
    )

    # 追加の文脈情報を活かした質問応答を実現
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",  # 回答の生成方法を調整
        retriever=retriever,
        return_source_documents=False,  # Trueなら参照した文脈情報をレスポンスに含め
        verbose=True
    )


def ask(qa, query):
    with get_openai_callback() as cb:
        # response = qa.run(query)
        # answer = response['result']
        response = qa(query)
        answer = response['result']

    return answer, cb.total_cost, cb.prompt_tokens, cb.completion_tokens


def pdf_ask_my_pdf():
    # st.title("PDF(s)独自QAサイト")
    llm = select_model()
    container = st.container()
    response_container = st.container()

    with container:
        query = st.text_input("Query: ", key="input")
        if not query:
            answer = None
        else:
            qa = build_qa_model(llm)
            if qa:
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost, prompt_token, answer_token = ask(qa, query)
                st.session_state.costs.append(cost)
                st.session_state.prompt_tokens.append(prompt_token)
                st.session_state.answer_tokens.append(answer_token)
            else:
                answer = None

        if answer:
            with response_container:
                st.markdown("## Answer")
                st.write(answer)


def main():
    init_page()
    # n_chars = select_n_chars()

    selection = st.sidebar.radio(
        "ページ", ["PDF(s)独自QAサイト", "チャットQAサイト", "Webページ要約サイト", "Youtube要約サイト"])
    if selection == "PDF(s)独自QAサイト":
        pdf_upload_and_build_vector_db()
        pdf_ask_my_pdf()

    elif selection == "チャットQAサイト":
        st.header("チャットQAサイト🗣️")
        init_messages()
        llm = select_model()
        # ユーザーの入力を監視
        if user_input := st.chat_input("聞きたいことを入力してね！"):
            st.session_state.messages.append(HumanMessage(content=user_input))
            with st.spinner("ChatGPT is typing ..."):
                answer, cost, prompt_token, answer_token = get_answer(
                    llm, st.session_state.messages)
            st.session_state.messages.append(AIMessage(content=answer))
            st.session_state.costs.append(cost)
            st.session_state.prompt_tokens.append(prompt_token)
            st.session_state.answer_tokens.append(answer_token)

        # チャット履歴の表示
        display_chat_history()

    elif selection == "Webページ要約サイト":
        st.header("Webページ要約サイト✏️")

    else:
        st.header("Youtube Summarizer Apps🎥")
        st.subheader("15分以上の動画を要約するときは、サイドバーで:blue[GPT-3.5-16k]を選択してね")

    costs = st.session_state.get('costs', [])
    prompt_tokens = st.session_state.get('prompt_tokens', [])
    answer_tokens = st.session_state.get('answer_tokens', [])

    # 料金計算
    calculate_costs()

    # Prompt Tokens, Completion Tokensの表示
    display_tokens()


if __name__ == "__main__":
    main()
