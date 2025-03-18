import os
import logging
from flask import Flask, render_template, request, jsonify
from chatbot import AphatorChatbot

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "default-secret-key-for-dev")

# Initialize chatbot
chatbot = AphatorChatbot()

@app.route('/')
def index():
    """Render the main chat interface."""
    return render_template('index.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    """Process user messages and return chatbot responses."""
    try:
        data = request.json
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Empty message'}), 400
        
        logger.debug(f"Received message: {user_message}")
        
        # Process user message through chatbot
        response = chatbot.get_response(user_message)
        
        return jsonify({
            'response': response
        })
    except Exception as e:
        logger.error(f"Error processing message: {str(e)}")
        return jsonify({'error': 'Failed to process your message. Please try again.'}), 500
