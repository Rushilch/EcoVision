"""
Groq Advisor using the Qwen3-32B Model
-------------------------------------
Generates environmental improvement suggestions
based on UHI, flood risk, and plastic waste risk.

Requirements:
  - pip install groq python-dotenv
  - .env file with GROQ_API_KEY at project root
"""

import os
from dotenv import load_dotenv
from groq import Groq

# Load the .env file
load_dotenv()

# Get Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise EnvironmentError(
        "GROQ_API_KEY not found in .env. Please set it to use Groq AI."
    )

# Initialize Groq client
client = Groq(api_key=GROQ_API_KEY)

def get_environment_suggestions(
    country: str,
    uhi: float,
    flood: float,
    plastic: float
) -> str:

    prompt = f"""
You are an environmental sustainability expert advising urban planners. dont give me your thinking process just the suggestions.

Context:
Country: {country}
Urban Heat Island (UHI) intensity: {uhi}
Flood risk probability: {flood}
Plastic waste risk: {plastic}

Task:
Provide 3–5 realistic, actionable measures to improve environmental conditions.
Rules:
- Focus on urban planning, infrastructure, and policy
- Do NOT describe AI internals
- Keep recommendations clear and concise
"""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",               # 🟢 Best supported model as per Groq docs
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,
        max_tokens=300
    )

    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    print(
        get_environment_suggestions(
            country="India",
            uhi=0.8,
            flood=0.65,
            plastic=0.55
        )
    )
