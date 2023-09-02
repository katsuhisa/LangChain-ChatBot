import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from urllib.parse import urlparse


def init_page():
    # ウェブページの設定
    st.set_page_config(
        page_title="Youtube Summarizer Apps",
        page_icon="🚀"
    )
    st.header("Youtube Summarizer Apps🎥")
    st.sidebar.title("Options")
    # st.session_state.model_name = "GPT-3.5-16k"
    st.session_state.costs = []
    st.session_state.prompt_tokens = []
    st.session_state.answer_tokens = []


def select_model():
    model = st.sidebar.radio(
        "今回使用するモデルは:",
        ["GPT-3.5", "GPT-3.5-16k", "GPT-4"],
        captions=["gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k-0613", "gpt-4"])
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

# クリアボタンを押したときに、メッセージ履歴初期化と同じ処理を走らせると、履歴を消す


def select_n_chars():
    # n_charsパラメータを追加
    n_chars = st.sidebar.slider(
        "要約する文字数:", min_value=100, max_value=1000, value=100, step=100)
    return n_chars


# def init_messages():
#     clear_button = st.sidebar.button("Clear Conversation", key="clear")
#     if clear_button or "messages" not in st.session_state:
#         st.session_state.messages = [
#             SystemMessage(content="箇条書きで返答してください.")
#         ]
#         st.session_state.costs = []
#         st.session_state.prompt_tokens = []
#         st.session_state.answer_tokens = []

# URL受け取り関数


def get_url_input():
    url = st.text_input("Youtube URL: ", key="input")
    return url

# URLが正しいか検証する関数


def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False


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


# Youtube動画の内容(正確には字幕)を取得. DocumentはLangChainに直接渡すことができる
def get_document(url):
    """
    # key | 型	| 説明
    # page_content | string | ドキュメントの生のテキストデータ
    # metadata | dict | テキストに関するメタデータを保存するためのキー/値ストア（ソースURL、著者など）
    """
    with st.spinner("内容読み取り中 ..."):
        loader = YoutubeLoader.from_youtube_url(
            url,
            add_video_info=True,  # タイトルや再生数も取得できる
            language=['en', 'ja']  # 英語→日本語の優先順位で字幕を取得
        )
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(  # Text分割の指示
            model_name=st.session_state.model_name,
            chunk_size=st.session_state.max_token,  # chunk(ブロック)１単位の文字数
            # chunk_overlap=st.session_state.overlap,  # chunk間での重複文字数。段落間で文脈を維持したい場合などに有効。
            chunk_overlap=0,  # chunk間での重複文字数。段落間で文脈を維持したい場合などに有効。
        )
        # load_and_splitで、長いDocumentの分割を自動で行なってくれる。
        return loader.load_and_split(text_splitter=text_splitter)

# get_documentで取得した内容を要約


def document_summarize(llm, docs):
    prompt_template = """Write a concise Japanese summary of the following transcript of Youtube Video.
    
    {text}

    - ここから日本語で書くこと
    - 必ず3段落以内の簡潔にまとめること:
    - 必ず200文字以内で簡潔にまとめること
    - 箇条書きで出力すること
    """

    PROMPT = PromptTemplate(template=prompt_template,
                            input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm,  # e.g. ChatOpenAI(temperature=0)
            chain_type="stuff",
            verbose=True,  # チェーンは実行中に詳細なログを出力
            prompt=PROMPT
        )
        response = chain({
            "input_documents": docs,
            "token_max": st.session_state.max_token  # 長いコンテキストを扱えるモデルを利用するようにするために必要
            # "overlap": st.session_state.overlap  # 長いコンテキストを扱えるモデルを利用するようにするために必要
        },
            return_only_outputs=True)
    return response['output_text'], cb.total_cost, cb.prompt_tokens, cb.completion_tokens


def main():
    init_page()
    llm = select_model()
    # n_chars = select_n_chars()
    # init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('　👆適切なURLを入力してください。')
            output_text = None
        else:
            if url:
                document = get_document(url)
                with st.spinner("ChatGPT is typing ..."):
                    output_text, cost, prompt_token, answer_token = document_summarize(
                        llm, document)
                st.session_state.costs.append(cost)
                st.session_state.prompt_tokens.append(prompt_token)
                st.session_state.answer_tokens.append(answer_token)
            else:
                output_text = None

    if output_text:
        with response_container:
            st.markdown("## Summary")
            # st.write("要約文字数：", n_chars)
            st.write(output_text)
            st.markdown("---")
            st.markdown("## Original Text")
            st.write(document)

    # 料金計算
    calculate_costs()

    # Prompt Tokens, Completion Tokensの表示
    display_tokens()


if __name__ == '__main__':
    main()
