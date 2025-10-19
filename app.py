"""
Fake News Detection Flask API
Provides endpoints for the web dashboard
"""

from flask import Flask, render_template, request, jsonify
import pickle
import re
import os

app = Flask(__name__)

# Load model and vectorizer
try:
    with open('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    print("✓ Model and vectorizer loaded successfully")
except FileNotFoundError:
    print("❌ Model files not found. Please run train_model.py first.")
    model = None
    vectorizer = None

def preprocess_text(text):
    """Clean and preprocess text data"""
    if not text:
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

@app.route('/')
def home():
    """Render the main dashboard"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if a news headline is fake or real"""
    try:
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get headline from request
        data = request.get_json()
        headline = data.get('headline', '')
        
        if not headline or len(headline.strip()) == 0:
            return jsonify({
                'error': 'Please provide a headline'
            }), 400
        
        # Preprocess
        cleaned_headline = preprocess_text(headline)
        
        if len(cleaned_headline) == 0:
            return jsonify({
                'error': 'Headline contains no valid text'
            }), 400
        
        # Vectorize
        headline_vectorized = vectorizer.transform([cleaned_headline])
        
        # Predict
        prediction = model.predict(headline_vectorized)[0]
        probability = model.predict_proba(headline_vectorized)[0]
        
        # Get confidence
        confidence = max(probability) * 100
        
        # Determine result
        is_fake = bool(prediction == 1)
        label = "FAKE NEWS" if is_fake else "REAL NEWS"
        
        return jsonify({
            'prediction': label,
            'is_fake': is_fake,
            'confidence': round(confidence, 2),
            'probabilities': {
                'real': round(probability[0] * 100, 2),
                'fake': round(probability[1] * 100, 2)
            }
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and vectorizer is not None
    })

if __name__ == '__main__':
    print("\n" + "="*60)
    print("FAKE NEWS DETECTION API")
    print("="*60)
    print("Starting Flask server...")
    print("Dashboard will be available at: http://localhost:5000")
    print("="*60 + "\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
