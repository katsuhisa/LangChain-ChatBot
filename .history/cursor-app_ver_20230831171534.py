import streamlit as st
from langchain import LangChain

# チャット履歴を保存するためのリストを作成します
chat_history = []

def main():
    # LangChainのインスタンスを作成します
    lc = LangChain()

    # StreamlitのUIを作成します
    st.title('LangChain ChatBot')

    # ユーザーからの入力を受け取るテキストボックスを作成します
    user_input = st.text_input('メッセージを入力してください')

    # ユーザーが何かを入力した場合、LangChainを使用して応答を生成します
    if user_input:
        response = lc.generate_response(user_input)
        # ユーザーのメッセージとLangChainの応答をチャット履歴に追加します
        chat_history.append({'user': user_input, 'bot': response})

    # チャット履歴を表示します
    for chat in chat_history:
        st.text('User: ' + chat['user'])
        st.text('Bot: ' + chat['bot'])

if __name__ == "__main__":
    main()