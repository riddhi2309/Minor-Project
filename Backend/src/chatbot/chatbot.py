import os
import sys

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

import ollama
from security_pipeline import SecurityPipeline   # ← only this, no blocking.py needed


def generate_response(prompt: str) -> str:
    response = ollama.chat(
        model="tinyllama",
        messages=[
            {"role": "system", "content": "You are a secure AI assistant."},
            {"role": "user",   "content": prompt},
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    print("🔐 Secure AI Chatbot Started (type 'exit' to quit)\n")

    pipeline = SecurityPipeline()

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye.")
            break

        # Step 1: analyze
        result = pipeline.analyze(user_input)

        # Step 2: block or allow
        is_blocked, message, log_id = pipeline.block(user_input, result)

        if is_blocked:
            print(message)
            print(f"   [log id: {log_id}]\n")
            continue          # LLM never called

        # Step 3: safe — call LLM
        reply = generate_response(user_input)
        print(f"Bot: {reply}\n")