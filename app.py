import streamlit as st
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import requests

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="Semantic AI Movie Agent",
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

st.title("🎬 Semantic AI Movie Agent")

st.write(
    "AI-powered entertainment recommendations using semantic similarity 🧠"
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
    default=["English", "Malayalam"]
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
# CANDIDATE CONTENT DATABASE
# ---------------------------------------------------

candidate_content = [

    {
        "title": "Dark",
        "description": "time travel mystery with existential philosophy and emotional complexity"
    },

    {
        "title": "Severance",
        "description": "psychological corporate thriller exploring identity and isolation"
    },

    {
        "title": "Arrival",
        "description": "emotionally intelligent sci-fi exploring communication and time"
    },

    {
        "title": "Joji",
        "description": "slow-burn Malayalam psychological crime drama about greed and loneliness"
    },

    {
        "title": "The Bear",
        "description": "emotionally intense storytelling about trauma, pressure and purpose"
    },

    {
        "title": "Paatal Lok",
        "description": "dark Indian crime thriller exploring morality and social decay"
    },

    {
        "title": "Signal",
        "description": "Korean mystery thriller involving time communication and serial crimes"
    },

    {
        "title": "True Detective",
        "description": "philosophical crime investigation with psychological darkness"
    }
]

# ---------------------------------------------------
# OTT PLATFORM MAPPING
# ---------------------------------------------------

def get_platform(title):

    platform_map = {

        "Dark": "Netflix",
        "Severance": "Apple TV+",
        "Arrival": "Amazon Prime Video",
        "Joji": "Amazon Prime Video",
        "The Bear": "JioHotstar",
        "Paatal Lok": "Amazon Prime Video",
        "Signal": "Netflix",
        "True Detective": "JioHotstar"
    }

    return platform_map.get(title, "Search Online")

# ---------------------------------------------------
# OMDB HELPER
# ---------------------------------------------------

def get_movie_data(title):

    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"

    response = requests.get(url)

    return response.json()

# ---------------------------------------------------
# OPENAI EMBEDDINGS
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

generate = st.button("🎯 Generate Semantic Recommendations")

# ---------------------------------------------------
# MAIN RECOMMENDATION LOGIC
# ---------------------------------------------------

if generate:

    with st.spinner("Analyzing semantic taste profile..."):

        # Combine user taste + mood
        combined_taste = " ".join(user_taste) + " " + mood

        # Generate user embedding
        user_embedding = get_embedding(
            combined_taste
        )

        # Recommendation scoring
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
                    float(similarity) * 100,
                    2
                )
            })

        # Sort recommendations
        recommendations.sort(
            key=lambda x: x["similarity_score"],
            reverse=True
        )

        # Save in session state
        st.session_state.recommendations = recommendations

        st.session_state.generated = True

# ---------------------------------------------------
# DISPLAY UI
# ---------------------------------------------------

if st.session_state.generated:

    st.success("Semantic Recommendations Ready 🎬")

    cols = st.columns(2)

    for idx, item in enumerate(
        st.session_state.recommendations[:6]
    ):

        movie = get_movie_data(item["title"])

        with cols[idx % 2]:

            st.subheader(item["title"])

            poster = movie.get("Poster")

            if poster and poster != "N/A":
                st.image(poster)

            st.write(
                f"🔥 Semantic Match: "
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

            # ---------------------------------------------------
            # FEEDBACK BUTTONS
            # ---------------------------------------------------

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