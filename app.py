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
    default=["Malayalam"]
)

content_type = st.sidebar.multiselect(
    "Content Type",
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
# CANDIDATE CONTENT DATABASE
# ---------------------------------------------------

candidate_content = [

    {
        "title": "Joji",
        "language": "Malayalam",
        "type": "Movies",
        "description": "slow-burn Malayalam psychological crime drama about greed and loneliness"
    },

    {
        "title": "Kumbalangi Nights",
        "language": "Malayalam",
        "type": "Movies",
        "description": "emotionally rich Malayalam drama exploring relationships and masculinity"
    },

    {
        "title": "Bramayugam",
        "language": "Malayalam",
        "type": "Movies",
        "description": "dark atmospheric horror with psychological depth and folklore"
    },

    {
        "title": "Manjummel Boys",
        "language": "Malayalam",
        "type": "Movies",
        "description": "survival thriller based on friendship, fear and emotional resilience"
    },

    {
        "title": "Dark",
        "language": "English",
        "type": "TV Shows",
        "description": "time travel mystery with existential philosophy and emotional complexity"
    },

    {
        "title": "Severance",
        "language": "English",
        "type": "TV Shows",
        "description": "psychological corporate thriller exploring identity and isolation"
    },

    {
        "title": "Arrival",
        "language": "English",
        "type": "Movies",
        "description": "emotionally intelligent sci-fi exploring communication and time"
    },

    {
        "title": "The Bear",
        "language": "English",
        "type": "TV Shows",
        "description": "emotionally intense storytelling about trauma, pressure and purpose"
    },

    {
        "title": "Paatal Lok",
        "language": "Hindi",
        "type": "Web Series",
        "description": "dark Indian crime thriller exploring morality and social decay"
    },

    {
        "title": "Signal",
        "language": "Korean",
        "type": "TV Shows",
        "description": "Korean mystery thriller involving time communication and serial crimes"
    },

    {
        "title": "True Detective",
        "language": "English",
        "type": "TV Shows",
        "description": "philosophical crime investigation with psychological darkness"
    },

    {
        "title": "Attack on Titan",
        "language": "Japanese",
        "type": "Anime",
        "description": "dark fantasy anime exploring freedom, war and existential conflict"
    }
]

# ---------------------------------------------------
# OTT PLATFORM MAPPING
# ---------------------------------------------------

def get_platform(title):

    platform_map = {

        "Joji": "Amazon Prime Video",
        "Kumbalangi Nights": "Amazon Prime Video",
        "Bramayugam": "Sony LIV",
        "Manjummel Boys": "Disney+ Hotstar",
        "Dark": "Netflix",
        "Severance": "Apple TV+",
        "Arrival": "Amazon Prime Video",
        "The Bear": "JioHotstar",
        "Paatal Lok": "Amazon Prime Video",
        "Signal": "Netflix",
        "True Detective": "JioHotstar",
        "Attack on Titan": "Crunchyroll"
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
# OPENAI EMBEDDING FUNCTION
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

        combined_taste = " ".join(user_taste) + " " + mood

        user_embedding = get_embedding(
            combined_taste
        )

        recommendations = []

        for item in candidate_content:

            # Skip non-selected content types
            if item["type"] not in content_type:
                continue

            content_embedding = get_embedding(
                item["description"]
            )

            similarity = cosine_similarity(
                [user_embedding],
                [content_embedding]
            )[0][0]

            # ---------------------------------------------------
            # FINAL SCORE
            # ---------------------------------------------------

            final_score = similarity * 100

            # STRONG LANGUAGE PRIORITY
            if item["language"] in languages:

                final_score += 30

            else:

                final_score -= 20

            recommendations.append({

                "title": item["title"],
                "language": item["language"],
                "type": item["type"],
                "description": item["description"],
                "similarity_score": round(
                    float(final_score),
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
                f"🎬 Type: "
                f"{item['type']}"
            )

            st.write(
                f"🎭 Genre: "
                f"{movie.get('Genre', 'N/A')}"
            )

            st.write(
                f"🌍 Language: "
                f"{item['language']}"
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