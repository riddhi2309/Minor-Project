import re

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression

import pickle
import sys
import os

# Get project root (one level above chatbot folder)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

# Add project root to Python path
sys.path.append(PROJECT_ROOT)

# Now you can import from src
from security_pipeline import SecurityPipeline

# ========== CHATBOT ==========

import ollama



def generate_response(prompt):

    response = ollama.chat(

        model='tinyllama',

        messages=[

            {'role': 'system', 'content': 'You are a secure AI assistant.'},

            {'role': 'user', 'content': prompt}

        ]

    )

    return response['message']['content']



# ========== MAIN LOOP ==========

if __name__ == "__main__":
    print("🔐 Secure AI Chatbot Started (Type 'exit' to quit)\n")

    pipeline = SecurityPipeline()

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            break

        # 🔥 Run Security Check
        security_result = pipeline.analyze(user_input)

        if security_result["final_decision"] == "BLOCK":
            print("🚫 Blocked by AI Security System")
            print("Attack Type:", security_result["attack_type"])
            continue

        # ✅ If safe, generate response
        reply = generate_response(user_input)
        print("Bot:", reply)