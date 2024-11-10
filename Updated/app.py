import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Set your Gemini API key (replace with your actual key)
genai.configure(api_key="AIzaSyAlIzJLXc5aNJQ7a9doJcAzzUUuUSActJU")

# Model configuration
generation_config = {
    "temperature": 0.2,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 512, 
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-pro",
    generation_config=generation_config,
)

chat_session = model.start_chat()

def get_mental_health_conversation(user_question):
    try:
        with open("mental_health_resources.txt", "r") as f:
            mental_health_resources = f.read()
    except FileNotFoundError:
        return "Error: mental_health_resources.txt not found. Please make sure the file exists in the same directory."

    # Modified prompt for a more conversational tone without overloading information
    prompt = (
        "You are a mental health support assistant for Vaal University of Technology (VUT). "
        "Your role is to have an empathetic, supportive conversation with the student about their feelings, mental health, or situation. "
        "Ask questions to understand their experience better and offer a comforting, understanding response. Only provide detailed resources if the student expresses a need for help or asks specifically for support options at VUT. "
        "Focus on a human-like, compassionate conversation where the student feels heard. "
        "If needed, kindly suggest appropriate mental health resources after building rapport and understanding the situation. "
        f"\n\n{mental_health_resources}\n\nStudent's Question: {user_question}"
    )

    response = chat_session.send_message(prompt)
    return response.text

@app.route('/api/get_mental_health_support', methods=['POST'])
def api_get_mental_health_support():
    user_question = request.json.get('question')
    answer = get_mental_health_conversation(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
