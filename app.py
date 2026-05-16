import streamlit as st

st.set_page_config(
    page_title="AI Movie Agent",
    page_icon="🎬"
)

st.title("🎬 AI Movie Recommendation Agent")

st.success("Deployment Successful!")

st.write("Your AI app is now live 🚀")

mood = st.selectbox(
    "Choose your mood",
    [
        "Mind-bending",
        "Dark",
        "Emotional",
        "Funny"
    ]
)

st.write(f"You selected: {mood}")