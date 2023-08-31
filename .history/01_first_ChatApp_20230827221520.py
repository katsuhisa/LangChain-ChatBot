import streamlit as st

st.set_page_config(
    page_title="My Great ChatGPT",
    page_icon="🤗"
)
st.header("My Great ChatGPT 🤗")

if user_input := st.chat_input("聞きたいことを入力してね！"):
    # なにか入力されればここが実行される