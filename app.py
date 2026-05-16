import streamlit as st
from openai import OpenAI
import os

# Page config
st.set_page_config(
    page_title="AI Movie Agent",
    page_icon="🎬",
    layout="wide"
)

# OpenAI client
client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

# Title
st.title("🎬 AI Movie Recommendation Agent")

st.write("Your personal AI entertainment curator 🚀")

# Sidebar
st.sidebar.header("Preferences")

mood = st.sidebar.selectbox(
    "Choose your mood",
    [
        "Mind-bending",
        "Dark",
        "Emotional",
        "Thrilling",
        "Funny",
        "Feel Good"
    ]
)

language = st.sidebar.multiselect(
    "Preferred Languages",
    [
        "English",
        "Hindi",
        "Malayalam",
        "Kannada",
        "Tamil",
        "Korean"
    ],
    default=["English", "Malayalam"]
)

content_type = st.sidebar.multiselect(
    "Content Types",
    [
        "Movies",
        "TV Shows",
        "Web Series"
    ],
    default=["Movies", "TV Shows"]
)

# Button
generate = st.button("🎯 Generate Recommendations")

# AI Logic
if generate:

    with st.spinner("Analyzing your taste..."):

        prompt = f"""
        Recommend:
        - 5 movies
        - 5 TV shows/web-series

        User mood:
        {mood}

        Preferred languages:
        {language}

        Preferred content types:
        {content_type}

        IMPORTANT:
        - Include regional Indian content if suitable
        - Include international content if relevant
        - Avoid generic recommendations
        - Prioritize intelligent storytelling
        - Mention:
            1. title
            2. language
            3. genre
            4. short reason
        """

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        recommendations = response.choices[0].message.content

    st.success("Recommendations Ready 🎬")

    st.write(recommendations)