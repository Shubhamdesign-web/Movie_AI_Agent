import streamlit as st
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Dynamic Semantic AI Movie Agent",
    page_icon="🎬",
    layout="wide"
)

# ---------------------------------------------------
# API CONFIG
# ---------------------------------------------------

client = OpenAI(
    api_key=st.secrets["OPENAI_API_KEY"]
)

OMDB_API_KEY = st.secrets["OMDB_API_KEY"]

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "recommendations" not in st.session_state:
    st.session_state.recommendations = []

if "generated" not in st.session_state:
    st.session_state.generated = False

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------

st.title("🎬 Dynamic Semantic AI Movie Agent")

st.write(
    "AI-powered personalized entertainment discovery 🧠"
)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------

st.sidebar.header("🎭 Your Preferences")

mood = st.sidebar.selectbox(
    "Choose your mood",
    [
        "Mind-bending",
        "Dark",
        "Emotional",
        "Thrilling",
        "Funny",
        "Feel Good",
        "Philosophical",
        "Intense"
    ]
)

languages = st.sidebar.multiselect(
    "Preferred Languages",
    [
        "English",
        "Hindi",
        "Malayalam",
        "Kannada",
        "Tamil",
        "Telugu",
        "Korean",
        "Japanese"
    ],
    default=["Malayalam"]
)

content_type = st.sidebar.multiselect(
    "Content Types",
    [
        "Movies",
        "TV Shows",
        "Web Series",
        "Anime"
    ],
    default=["Movies", "TV Shows"]
)

# ---------------------------------------------------
# USER TASTE PROFILE
# ---------------------------------------------------

user_taste = [
    "mind-bending sci-fi",
    "psychological thrillers",
    "slow-burn mystery",
    "philosophical storytelling",
    "emotionally intelligent narratives",
    "dark crime drama",
    "existential themes"
]

# ---------------------------------------------------
# OMDB HELPER
# ---------------------------------------------------

def get_movie_data(title):

    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"

    response = requests.get(url)

    return response.json()

# ---------------------------------------------------
# OTT PLATFORM MAPPING
# ---------------------------------------------------

def get_platform(title):

    platform_map = {

        "Dark": "Netflix",
        "Severance": "Apple TV+",
        "Joji": "Amazon Prime Video",
        "Kumbalangi Nights": "Amazon Prime Video",
        "Bramayugam": "Sony LIV",
        "Manjummel Boys": "Disney+ Hotstar",
        "Paatal Lok": "Amazon Prime Video",
        "Signal": "Netflix",
        "True Detective": "JioHotstar"
    }

    return platform_map.get(title, "Search Online")

# ---------------------------------------------------
# EMBEDDING FUNCTION
# ---------------------------------------------------

def get_embedding(text):

    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding

# ---------------------------------------------------
# GENERATE BUTTON
# ---------------------------------------------------

generate = st.button("🎯 Generate Dynamic Recommendations")

# ---------------------------------------------------
# MAIN LOGIC
# ---------------------------------------------------

if generate:

    with st.spinner("Generating personalized recommendations..."):

        # ---------------------------------------------------
        # DYNAMIC OPENAI RECOMMENDATION GENERATION
        # ---------------------------------------------------

        prompt = f"""
        You are an elite movie and TV recommendation AI.

        USER MOOD:
        {mood}

        USER LANGUAGE PREFERENCES:
        {languages}

        CONTENT TYPES:
        {content_type}

        USER TASTE:
        {user_taste}

        Generate 15 highly personalized recommendations.

        IMPORTANT:
        - Prioritize preferred languages strongly
        - Avoid generic recommendations
        - Include regional Indian cinema if relevant
        - Include TV shows/web-series if relevant
        - Include hidden gems
        - Recommendations should vary significantly based on mood and language

        Return ONLY:
        title | short thematic description

        Example:
        Dark | philosophical sci-fi mystery involving time travel
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

        result = response.choices[0].message.content

        lines = result.split("\n")

        candidate_content = []

        for line in lines:

            if "|" in line:

                parts = line.split("|")

                if len(parts) >= 2:

                    candidate_content.append({

                        "title": parts[0].strip(),

                        "description": parts[1].strip()
                    })

        # ---------------------------------------------------
        # SEMANTIC RANKING
        # ---------------------------------------------------

        combined_taste = " ".join(user_taste) + " " + mood

        user_embedding = get_embedding(
            combined_taste
        )

        recommendations = []

        for item in candidate_content:

            content_embedding = get_embedding(
                item["description"]
            )

            similarity = cosine_similarity(
                [user_embedding],
                [content_embedding]
            )[0][0]

            recommendations.append({

                "title": item["title"],

                "description": item["description"],

                "similarity_score": round(
                    float(similarity * 100),
                    2
                )
            })

        # Sort recommendations
        recommendations.sort(
            key=lambda x: x["similarity_score"],
            reverse=True
        )

        st.session_state.recommendations = recommendations

        st.session_state.generated = True

# ---------------------------------------------------
# DISPLAY UI
# ---------------------------------------------------

if st.session_state.generated:

    st.success("Recommendations Ready 🎬")

    cols = st.columns(2)

    for idx, item in enumerate(
        st.session_state.recommendations[:8]
    ):

        movie = get_movie_data(item["title"])

        with cols[idx % 2]:

            st.subheader(item["title"])

            poster = movie.get("Poster")

            if poster and poster != "N/A":
                st.image(poster)

            st.write(
                f"🔥 Match Score: "
                f"{item['similarity_score']}%"
            )

            st.write(
                f"🎭 Genre: "
                f"{movie.get('Genre', 'N/A')}"
            )

            st.write(
                f"🌍 Language: "
                f"{movie.get('Language', 'N/A')}"
            )

            st.write(
                f"⭐ IMDb Rating: "
                f"{movie.get('imdbRating', 'N/A')}"
            )

            platform = get_platform(item["title"])

            st.write(
                f"📺 Watch On: {platform}"
            )

            st.write("### 🧠 Why It Matches")

            st.write(
                item["description"]
            )

            col1, col2 = st.columns(2)

            with col1:

                if st.button(
                    "👍 Interested",
                    key=f"like_{idx}"
                ):

                    st.success(
                        f"You liked {item['title']}!"
                    )

            with col2:

                if st.button(
                    "👎 Skip",
                    key=f"dislike_{idx}"
                ):

                    st.warning(
                        f"You skipped {item['title']}"
                    )

            st.divider()