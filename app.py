from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        uploaded_files = data.get('uploaded_files', [])
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400
        
        # Create contextual prompt
        contextual_prompt = f"""You are a Legal AI Assistant. Provide helpful, accurate legal information while always emphasizing that your responses are for informational purposes only and not legal advice.

User's question: {message}"""

        if uploaded_files:
            file_names = [f['name'] for f in uploaded_files]
            contextual_prompt += f"\n\nNote: The user has uploaded: {', '.join(file_names)}."

        chat_completion = groq_client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": """You are a helpful Legal AI Assistant. Provide informative responses about legal matters while always clarifying that this is general information only, not legal advice. 

IMPORTANT FORMATTING RULES:
- Start with a brief introduction paragraph
- Use clear section headings followed by a colon (:)
- Under each section, use bullet points (•) for key information
- Use numbered lists (1., 2., 3.) for step-by-step processes or categories
- Keep bullet points concise and focused
- Always end with a disclaimer paragraph
- Structure your response like this example:

Introduction paragraph explaining the topic.

Key Points:
• First important point
• Second important point
• Third important point

Legal Consequences:
1. First consequence with details
2. Second consequence with details
3. Third consequence with details

Always recommend consulting with a qualified attorney for specific legal situations."""
        },
                {
                    "role": "user",
                    "content": contextual_prompt
                }
            ],
            model="llama3-8b-8192",
            temperature=0.7,
            max_tokens=1000
        )

        response_content = chat_completion.choices[0].message.content

        return jsonify({
            'response': response_content,
            'model_used': 'llama3-8b-8192'
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(debug=True, port=8000)