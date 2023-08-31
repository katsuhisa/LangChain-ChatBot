import streamlit as st

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,  # システムメッセージ
    HumanMessage,  # 人間の質問
    AIMessage  # ChatGPTの返答
)

def main():
    llm = ChatOpenAI( # ChatGPT APIを呼んでくれる機能
        temperature=0.5 # 生成テキストの"ランダム性"や"多様性"を制御。0〜1までが一般的。高いほどクリエイティブに。
        )  
    """
    他のLLMを呼び出す場合の例
    # Azure版 ChatGPT
    llm = AzureChatOpenAI(
        openai_api_base=BASE_URL,
        openai_api_version="2023-03-15-preview",
        deployment_name=DEPLOYMENT_NAME,
        openai_api_key=API_KEY,
        openai_api_type="azure",
    )
    # Google PaLM
    llm = ChatVertexAI()
    """
    
    # ウェブページの設定
    st.set_page_config(
        page_title="My First ChatGPT Apps",
        page_icon="🤗"
    )
    st.header("My First ChatGPT Apps🤗")

    # チャット履歴の初期化
    if "messages" not in st.session_state:
        st.session_state.messages = [
            SystemMessage(content="絶対に関西弁で返答してください.")
        ]

    # ユーザーの入力を監視
    if user_input := st.chat_input("聞きたいことを入力してね！"):
        st.session_state.messages.append(HumanMessage(content=user_input))
        
        # Streamlitのスピナー機能を利用
        with st.spinner("ChatGPT is typing ..."):
            response = llm(st.session_state.messages)
        st.session_state.messages.append(AIMessage(content=response.content))

    # チャット履歴の表示
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


if __name__ == '__main__':
    main()


