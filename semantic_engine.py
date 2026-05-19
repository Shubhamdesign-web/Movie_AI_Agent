from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# OpenAI client
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
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
# CANDIDATE CONTENT
# ---------------------------------------------------

candidate_content = [
    {
        "title": "Severance",
        "description": "psychological corporate mystery with existential themes"
    },

    {
        "title": "Dark",
        "description": "time travel mystery with philosophical and emotional depth"
    },

    {
        "title": "Joji",
        "description": "slow-burn psychological crime drama exploring greed and isolation"
    },

    {
        "title": "Arrival",
        "description": "emotionally intelligent sci-fi exploring time and communication"
    },

    {
        "title": "The Bear",
        "description": "intense emotional storytelling about trauma and purpose"
    }
]

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
# CREATE USER TASTE VECTOR
# ---------------------------------------------------

combined_taste = " ".join(user_taste)

user_embedding = get_embedding(combined_taste)

# ---------------------------------------------------
# SEMANTIC SIMILARITY
# ---------------------------------------------------

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
        "similarity_score": round(float(similarity) * 100, 2)
    })

# ---------------------------------------------------
# SORT RESULTS
# ---------------------------------------------------

recommendations.sort(
    key=lambda x: x["similarity_score"],
    reverse=True
)

# ---------------------------------------------------
# OUTPUT
# ---------------------------------------------------

print("\n🎬 SEMANTIC RECOMMENDATIONS\n")

for item in recommendations:

    print(
        f"{item['title']} → "
        f"{item['similarity_score']}% match"
    )