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
        page_title="チャットQAアプリ",
        page_icon="🤗"
    )
    st.header("🗣️AIチャットツール")
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


def init_messages():
    clear_button = st.sidebar.button("Clear Conversation", key="clear")
    if clear_button or "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="絶対に関西弁で返答してください.")
        ]
        st.session_state.costs = []
        st.session_state.prompt_tokens = []
        st.session_state.answer_tokens = []


def get_answer(llm, messages):  # 回答結果、コスト・プロンプト消費量の算出を行う関数
    with get_openai_callback() as cb:
        answer = llm(messages)
    return answer.content, cb.total_cost, cb.prompt_tokens, cb.completion_tokens


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


USD_JPY = 146.20  # 為替レートを定義します。この値は現在の為替レートによって変わります。


def calculate_costs(USD_JPY):
    costs = st.session_state.get('costs', [])
    total_cost_jpy = sum(costs) * USD_JPY  # 合計コストを日本円に換算します。
    st.sidebar.markdown("## Costs")
    # 換算した合計コストを表示します。
    st.sidebar.markdown(f"**Total cost: ¥{total_cost_jpy:.2f}**")
    for cost in costs:
        cost_jpy = cost * USD_JPY  # 各コストを日本円に換算します。
        st.sidebar.markdown(f"- ¥{cost_jpy:.2f}")  # 換算した各コストを表示します。


def display_tokens():
    prompt_tokens = st.session_state.get('prompt_tokens', [])
    answer_tokens = st.session_state.get('answer_tokens', [])
    st.sidebar.markdown("## Tokens")
    st.sidebar.markdown(f"Prompt Tokens: {prompt_tokens}")
    st.sidebar.markdown(f"Completion Tokens: {answer_tokens}")


def main():
    init_page()
    llm = select_model()
    init_messages()

    # ユーザーの入力を監視
    if user_input := st.chat_input("AIチャットくんに聞きたいことを入力してください！！"):
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

    # 料金計算
    calculate_costs(USD_JPY)

    # Prompt Tokens, Completion Tokensの表示
    display_tokens()


if __name__ == '__main__':
    main()
