import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

key = os.getenv("GROQ_API_KEY", "")
if not key:
    print("ERROR: GROQ_API_KEY not set in .env")
    exit(1)

client = Groq(api_key=key)
resp = client.chat.completions.create(
    model="llama-3.1-8b-instant",
    messages=[{"role": "user", "content": "Reply with exactly: GROQ_OK"}],
    temperature=0,
    max_tokens=10,
)
print("Groq response:", resp.choices[0].message.content.strip())
print("Model used   :", resp.model)
print("Tokens used  :", resp.usage.total_tokens)
