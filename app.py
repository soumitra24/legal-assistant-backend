from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from groq import Groq
import os
import logging
from dotenv import load_dotenv

# Configure logging for production
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Update CORS for production
allowed_origins = [
    "http://localhost:3000",  # Keep for local development
    "https://legal-chatbot-frontend.onrender.com"
]

# Add production domains from environment
if os.getenv("ALLOWED_ORIGINS"):
    production_origins = os.getenv("ALLOWED_ORIGINS").split(",")
    allowed_origins.extend(production_origins)

CORS(app, origins=allowed_origins)

# Initialize Groq client with error handling
try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Groq client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize Groq client: {e}")
    groq_client = None

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        # Check if Groq client is available
        if not groq_client:
            logger.error("Groq client not initialized")
            return jsonify({'error': 'AI service unavailable'}), 503

        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400

        message = data.get('message', '').strip()
        uploaded_files = data.get('uploaded_files', [])
        user_id = data.get('user_id', 'anonymous')  # For logging purposes
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        # Log the request (without sensitive data)
        logger.info(f"Chat request from user {user_id[:8] if len(user_id) > 8 else user_id}... - Message length: {len(message)}")
        
        # Create contextual prompt
        contextual_prompt = f"""You are a Legal AI Assistant. Provide helpful, accurate legal information while always emphasizing that your responses are for informational purposes only and not legal advice.

User's question: {message}"""

        if uploaded_files:
            file_names = [f['name'] for f in uploaded_files if 'name' in f]
            if file_names:
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
        
        # Log successful response
        logger.info(f"Successfully generated response for user {user_id[:8] if len(user_id) > 8 else user_id}...")

        return jsonify({
            'response': response_content,
            'model_used': 'llama3-8b-8192'
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': 'An error occurred while processing your request. Please try again.'
        }), 500

@app.route('/api/health', methods=['GET'])
def health():
    try:
        # Check if all services are working
        status = {
            'status': 'healthy',
            'groq_client': groq_client is not None,
            'environment': os.getenv('FLASK_ENV', 'development')
        }
        
        if groq_client is None:
            status['status'] = 'degraded'
            return jsonify(status), 503
            
        return jsonify(status), 200
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({'status': 'unhealthy', 'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Use environment variables for production
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("FLASK_ENV") == "development"
    app.run(debug=debug, host="0.0.0.0", port=port)