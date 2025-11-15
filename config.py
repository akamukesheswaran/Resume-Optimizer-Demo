"""
Configuration file for Resume Matcher Application
Place your Claude API key here
"""

# ============================================================================
# CLAUDE API CONFIGURATION
# ============================================================================
# Get your API key from: https://console.anthropic.com/
#CLAUDE_API_KEY = INSERT_YOUR_CLAUDE_API_KEY
# Claude model to use
CLAUDE_MODEL = "claude-sonnet-4-20250514"

# ============================================================================
# APPLICATION SETTINGS
# ============================================================================
# Maximum file upload size (in bytes) - 5MB
MAX_FILE_SIZE = 5 * 1024 * 1024

# Allowed file extensions for resume upload
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Upload folder
UPLOAD_FOLDER = 'uploads'

# Score thresholds
STRONG_MATCH_THRESHOLD = 0.80
GOOD_MATCH_THRESHOLD = 0.60

# ============================================================================
# MODEL PATHS
# ============================================================================
MODEL_DIR = 'models'
LABEL_ENCODER_PATH = f'{MODEL_DIR}/label_encoder.pkl'
SVM_MODEL_PATH = f'{MODEL_DIR}/svm_combined.pkl'
TFIDF_VECTORIZER_PATH = f'{MODEL_DIR}/tfidf_vectorizer_combined.pkl'

# ============================================================================
# AI OPTIMIZATION SETTINGS
# ============================================================================
# Maximum tokens for Claude response
MAX_TOKENS = 4000

# Temperature for AI responses (0.0 = deterministic, 1.0 = creative)
TEMPERATURE = 0.7