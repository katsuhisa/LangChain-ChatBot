import streamlit as st
# from streamlit_chat import message
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    HumanMessage,
    AIMessage
)
from langchain.callbacks import get_openai_callback

import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse


def init_page():
    # ウェブページの設定
    st.set_page_config(
        page_title="Webサイト要約ツール",
        page_icon="🤗"
    )
    st.header("📄Webサイト要約ツール")
    st.sidebar.title("Options")


def select_model():
    model = st.sidebar.selectbox(
        "今回使用するモデルは:",
        ("GPT-3.5", "GPT-3.5-16k"))
    if model == "GPT-3.5":
        model_name = "gpt-3.5-turbo-0613"
    else:
        model_name = "gpt-3.5-turbo-16k-0613"

    # サイドバーにスライダーを追加し、temperatureを0から2までの範囲で選択可能にする
    # 初期値は0.0、刻み幅は0.1とする
    temperature = st.sidebar.slider(
        "回答内容の創造性:", min_value=0.0, max_value=1.0, value=0.0, step=0.01)

    return ChatOpenAI(temperature=temperature, model_name=model_name)

# クリアボタンを押したときに、メッセージ履歴初期化と同じ処理を走らせると、履歴を消す


def select_n_chars():
    # n_charsパラメータを追加
    n_chars = st.sidebar.slider(
        "要約する文字数:", min_value=100, max_value=1000, value=100, step=100)
    return n_chars


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="箇条書きで返答してください.")
        ]
        st.session_state.costs = []
        st.session_state.prompt_tokens = []
        st.session_state.answer_tokens = []

# URL受け取り関数


def get_url_input():
    url = st.text_input("URL: ", key="input")
    return url

# URLが正しいか検証する関数


def validate_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

# 回答結果、コスト・プロンプト消費量の算出を行う関数


def get_answer(llm, messages):
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost, cb.prompt_tokens, cb.completion_tokens


USD_JPY = 146.20  # 為替レートを定義します。この値は現在の為替レートによって変わります。


def calculate_costs(USD_JPY):
    costs = st.session_state.get('costs', [])
    total_cost_jpy = sum(costs) * USD_JPY  # 合計コストを日本円に換算します。
    st.sidebar.markdown("## Costs")
    # 換算した合計コストを表示します。
    st.sidebar.markdown(f"**Total cost: {total_cost_jpy:.2f}円**")
    for i, cost in enumerate(costs, start=1):
        cost_jpy = cost * USD_JPY  # 各コストを日本円に換算します。
        st.sidebar.markdown(f"{i}回目: {cost_jpy:.2f}円")  # 換算した各コストを表示します。


def display_tokens():
    prompt_tokens = st.session_state.get('prompt_tokens', [])
    answer_tokens = st.session_state.get('answer_tokens', [])
    st.sidebar.markdown("## Tokens")
    st.sidebar.markdown(f"Prompt Tokens: {prompt_tokens}")
    st.sidebar.markdown(f"Completion Tokens: {answer_tokens}")


# ページの内容取得
def get_content(url):
    try:
        with st.spinner("記事読み込み中 ..."):
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            # fetch text from main (change the below code to filter page)
            # main, article, bodyタグを取得
            if soup.main:
                return soup.main.get_text()
            elif soup.article:
                return soup.article.get_text()
            else:
                return soup.body.get_text()
    except:
        st.write('something wrong')
        return None


# 要約指示Promptの構築. コンテンツが長い場合は先頭1000文字を利用して、適当に省略してる
def build_prompt(content, n_chars):
    return f"""
            以下はとあるWebページのコンテンツです。内容を以下の条件を守ってわかりやすく要約してください。
            - 要約は{n_chars}文字で。
            - 箇条書き
            - 関西弁
            - 中学生向け
            ========

            {content[:1000]}
            ========
            """


def main():
    init_page()
    llm = select_model()
    n_chars = select_n_chars()
    init_messages()

    st.divider()
    st.subheader("Prompt内容")
    code = """
    - 要約は300文字で。
    - 箇条書き
    - 関西弁
    - 中学生向け
    """
    st.code(code, language="textile")
    st.divider()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.subheader('  👆WebページのURLを入力してください。')

            answer = None
        else:
            content = get_content(url)
            if content:
                prompt = build_prompt(content, n_chars)
                st.session_state.messages.append(HumanMessage(content=prompt))
                with st.spinner("ChatGPT is typing ..."):
                    answer, cost, prompt_token, answer_token = get_answer(
                        llm, st.session_state.messages)
                # st.session_state.messages.append(AIMessage(content=answer))
                st.session_state.costs.append(cost)
                st.session_state.prompt_tokens.append(prompt_token)
                st.session_state.answer_tokens.append(answer_token)
            else:
                answer = None

    if answer:
        with response_container:
            st.markdown("## Summary")
            st.write("要約文字数：", n_chars)
            st.write(answer)
            # st.markdown("---")
            # st.markdown("## Original Text")
            # st.write(content)

    # 料金計算
    calculate_costs(USD_JPY)

    # Prompt Tokens, Completion Tokensの表示
    display_tokens()


if __name__ == '__main__':
    main()
