import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback

from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders import YoutubeLoader
from urllib.parse import urlparse


def init_page():
    # ウェブページの設定
    st.set_page_config(
        page_title="Youtube Summarizer Apps",
        page_icon="🤗"
    )
    st.header("Youtube Summarizer Apps🤗")
    st.sidebar.title("Options")


def select_model():
    model = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo-0613"
    else:
        model_name = "gpt-4"

    # サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    temperature = st.sidebar.slider(
        "Temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    return ChatOpenAI(temperature=temperature, model_name=model_name)

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
        return loader.load()  # Document

# get_documentで取得した内容を要約


def document_summarize(llm, docs):
    prompt_template = """Write a concise Japanese summary of the following transcript of Youtube Video.
    
    {text}

    ここから日本語で書いてね。必ず3段落以内の200文字以内で簡潔にまとめること:
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["text"])

    with get_openai_callback() as cb:
        chain = load_summarize_chain(
            llm,  # e.g. ChatOpenAI(temperature=0)
            chain_type="stuff",
            verbose=True,  # チェーンは実行中に詳細なログを出力
            prompt=PROMPT
        )
        response = chain({"input_documents": docs}, return_only_outputs=True)
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
            st.write('URLが無効です。適切なURLを入力してください。')
            answer = None
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
