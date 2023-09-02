import streamlit as st
from streamlit_chat import message
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
        page_title="Web Summarizer Apps",
        page_icon="🤗"
    )
    st.header("Web Summarizer Apps🤗")
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
        "Temperature:", min_value=0.0, max_value=2.0, value=0.0, step=0.01)

    return ChatOpenAI(temperature=temperature, model_name=model_name)

# クリアボタンを押したときに、メッセージ履歴初期化と同じ処理を走らせると、履歴を消す


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


def build_prompt(content):
    n_chars = st.sidebar.slider(
        "Temperature:", min_value=100, max_value=500, value=0.0, step=50)

    return f"""
            以下はとあるWebページのコンテンツです。内容を{n_chars}程度でわかりやすく要約してください。

            ========

            {content[:1000]}

            ========

            日本語で書いてね！
            """


def main():
    init_page()
    llm = select_model()
    init_messages()

    container = st.container()
    response_container = st.container()

    with container:
        url = get_url_input()
        is_valid_url = validate_url(url)
        if not is_valid_url:
            st.write('URLが無効です。適切なURLを入力してください。')
            answer = None
        else:
            content = get_content(url)
            if content:
                prompt = build_prompt(content)
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
            st.write(answer)
            # st.markdown("---")
            # st.markdown("## Original Text")
            # st.write(content)

    # 料金計算
    calculate_costs()

    # Prompt Tokens, Completion Tokensの表示
    display_tokens()


if __name__ == '__main__':
    main()
