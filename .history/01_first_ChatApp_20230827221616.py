import streamlit as st

# ウェブページの設定
st.set_page_config(
    page_title="My First ChatGPT Apps",
    page_icon="🤗"
)
st.header("My First ChatGPT Apps🤗")

if user_input := st.chat_input("聞きたいことを入力してね！"):
    # なにか入力されればここが実行される