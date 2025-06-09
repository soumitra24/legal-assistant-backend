from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import os
import logging
from dotenv import load_dotenv

# Configure logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

app = Flask(__name__)

# Update CORS for production
allowed_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

if os.getenv("ALLOWED_ORIGINS"):
    production_origins = [origin.strip() for origin in os.getenv("ALLOWED_ORIGINS").split(",")]
    allowed_origins.extend(production_origins)
    logger.info(f"Added production origins: {production_origins}")

logger.info(f"Allowed CORS origins: {allowed_origins}")
CORS(app, origins=allowed_origins, supports_credentials=True)

# Initialize Groq client with error handling
groq_client = None
try:
    from groq import Groq
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable not set")
    else:
        # Try different initialization approaches
        try:
            groq_client = Groq(api_key=api_key)
            logger.info("Groq client initialized successfully with standard method")
        except TypeError as e:
            logger.warning(f"Standard initialization failed: {e}")
            try:
                # Alternative initialization without extra kwargs
                groq_client = Groq(api_key=api_key)
                logger.info("Groq client initialized with alternative method")
            except Exception as e2:
                logger.error(f"Alternative initialization also failed: {e2}")
                groq_client = None
except ImportError as e:
    logger.error(f"Failed to import Groq: {e}")
    groq_client = None
except Exception as e:
    logger.error(f"Unexpected error initializing Groq client: {e}")
    groq_client = None

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'message': 'Legal Chatbot Backend API',
        'status': 'running',
        'port': os.getenv('PORT', '8000'),
        'groq_status': 'initialized' if groq_client else 'failed',
        'endpoints': {
            'health': '/api/health',
            'chat': '/api/chat (POST)'
        }
    })

@app.route('/api/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy' if groq_client else 'degraded',
        'groq_client': groq_client is not None,
        'environment': os.getenv('FLASK_ENV', 'development'),
        'port': os.getenv('PORT', '8000'),
        'groq_api_key_set': bool(os.getenv("GROQ_API_KEY"))
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        if not groq_client:
            logger.error("Groq client not initialized")
            return jsonify({'error': 'AI service unavailable - Groq client not initialized'}), 503

        data = request.get_json()
        if not data:
            return jsonify({'error': 'Invalid JSON data'}), 400

        message = data.get('message', '').strip()
        uploaded_files = data.get('uploaded_files', [])
        user_id = data.get('user_id', 'anonymous')
        
        if not message:
            return jsonify({'error': 'Message is required'}), 400

        logger.info(f"Chat request from user {user_id[:8] if len(user_id) > 8 else user_id}...")
        
        contextual_prompt = f"""You are a Legal AI Assistant. Provide helpful, accurate legal information while always emphasizing that your responses are for informational purposes only and not legal advice.

User's question: {message}"""

        if uploaded_files:
            file_names = [f['name'] for f in uploaded_files if 'name' in f]
            if file_names:
                contextual_prompt += f"\n\nNote: The user has uploaded: {', '.join(file_names)}."

        # Create chat completion
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
        logger.info(f"Successfully generated response for user {user_id[:8] if len(user_id) > 8 else user_id}...")

        return jsonify({
            'response': response_content,
            'model_used': 'llama3-8b-8192'
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'error': f'An error occurred while processing your request: {str(e)}'
        }), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8000))
    debug = os.getenv("FLASK_ENV") == "development"
    logger.info(f"Starting server on port {port}")
    logger.info(f"Groq client status: {'Ready' if groq_client else 'Not initialized'}")
    app.run(debug=debug, host="0.0.0.0", port=port)