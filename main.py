import os
import logging
from src.flask_app import app

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    # Start Flask server - Allow external connections
    host = '0.0.0.0'  # Accept connections from any IP
    port = int(os.getenv('FLASK_PORT', 5000))
    
    logger.info(f"Starting Face Recognition Service on {host}:{port}")
    
    # Run without SSL for development
    app.run(host=host, port=port, debug=False)
    
