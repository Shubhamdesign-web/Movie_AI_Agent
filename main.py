from openai import OpenAI
from dotenv import load_dotenv
import requests
import os
import json

# Load environment variables
load_dotenv()
print(os.getenv("OPENAI_API_KEY"))
# API Keys
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OMDB_API_KEY = os.getenv("OMDB_API_KEY")

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Load user taste profile
with open("watched.json", "r") as file:
    watched_data = json.load(file)

liked = watched_data["liked"]
disliked = watched_data["disliked"]
preferences = watched_data["preferences"]

# OMDb Search Function
def get_movie_data(title):

    url = f"http://www.omdbapi.com/?t={title}&apikey={OMDB_API_KEY}"

    response = requests.get(url)

    return response.json()

# Fetch metadata
movie_metadata = []

for item in liked:

    data = get_movie_data(item["title"])

    if data.get("Response") == "True":

        movie_metadata.append({
            "title": data.get("Title"),
            "genre": data.get("Genre"),
            "plot": data.get("Plot"),
            "imdb_rating": data.get("imdbRating"),
            "language": data.get("Language"),
            "type": data.get("Type")
        })

# Create AI Prompt
prompt = f"""
You are an elite movie and TV recommendation AI.

User Preferences:
{json.dumps(preferences, indent=2)}

Liked Content:
{json.dumps(liked, indent=2)}

Real Metadata:
{json.dumps(movie_metadata, indent=2)}

Disliked Content:
{json.dumps(disliked, indent=2)}

Recommend:
- 5 movies
- 5 TV shows/web-series

Requirements:
- Include regional cinema if suitable
- Include international content if relevant
- Prioritize intelligent storytelling
- Avoid generic mainstream recommendations
- Mention language and content type
- Explain why each recommendation matches
"""

try:

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    print("\n🎬 AI CURATED RECOMMENDATIONS\n")

    print(response.choices[0].message.content)

except Exception as e:

    print("\nError:")
    print(e)