# 📚 Comprehensive Documentation - Fake News Detection Dashboard

## Table of Contents
1. [Project Overview](#project-overview)
2. [What We Built](#what-we-built)
3. [Technology Stack Explained](#technology-stack-explained)
4. [Development Process](#development-process)
5. [Dataset Preparation](#dataset-preparation)
6. [Machine Learning Model Explained](#machine-learning-model-explained)
7. [Code Walkthrough](#code-walkthrough)
8. [API Documentation](#api-documentation)
9. [Frontend Architecture](#frontend-architecture)
10. [Deployment Guide](#deployment-guide)

---

## 1. Project Overview

### 1.1 What is Fake News Detection?

Fake news detection is a natural language processing (NLP) task that involves automatically classifying news articles or headlines as either **real** (authentic, verified) or **fake** (misleading, false, or manipulated). 

### 1.2 Our Approach

We built a web-based dashboard that uses machine learning to analyze news headlines and predict their authenticity. The system:
- Analyzes word patterns and frequencies in headlines
- Uses a trained Logistic Regression model
- Provides confidence scores for predictions
- Offers a user-friendly web interface

### 1.3 Project Goals

✅ Create an accessible web interface for fake news detection  
✅ Train a model on real datasets (Fake.csv and True.csv)  
✅ Achieve high accuracy through proper feature engineering  
✅ Provide transparency with confidence scores  
✅ Make it educational and easy to understand  

---

## 2. What We Built

### 2.1 Project Components

#### A. **Backend (Flask API)**
- REST API that serves predictions
- Model loading and management
- Text preprocessing pipeline
- Health check endpoints

#### B. **Machine Learning Model**
- CountVectorizer for feature extraction
- Logistic Regression for classification
- Training script with evaluation metrics
- Model persistence (saving/loading)

#### C. **Frontend (Web Dashboard)**
- Modern, responsive UI with gradient design
- Real-time prediction interface
- Confidence score visualization
- Example headlines for testing

#### D. **Documentation & Setup**
- Comprehensive README
- Requirements file
- Contributing guidelines
- MIT License

### 2.2 File Structure Explained

```
fake-news/
│
├── app.py                      # Flask backend server
├── train_model.py              # ML model training script
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
├── DOCUMENTATION.md            # This file - comprehensive guide
├── CONTRIBUTING.md             # Contribution guidelines
├── LICENSE                     # MIT License
├── .gitignore                 # Files to exclude from Git
│
├── templates/
│   └── index.html             # Web dashboard UI
│
├── Fake.csv                   # Fake news dataset (not in Git)
├── True.csv                   # Real news dataset (not in Git)
├── fake_news_model.pkl        # Trained model (generated)
└── vectorizer.pkl             # Fitted vectorizer (generated)
```

---

## 3. Technology Stack Explained

### 3.1 Python Libraries

#### **Flask (Web Framework)**
- **Purpose**: Creates the web server and API endpoints
- **Why Flask?**: Lightweight, easy to learn, perfect for ML projects
- **What it does**: Handles HTTP requests, serves HTML, processes predictions

#### **scikit-learn (Machine Learning)**
- **Purpose**: Provides ML algorithms and tools
- **Components we use**:
  - `CountVectorizer`: Converts text to numerical features
  - `LogisticRegression`: Classification algorithm
  - `train_test_split`: Splits data for training/testing
  - Evaluation metrics: accuracy, precision, recall, F1-score

#### **pandas (Data Processing)**
- **Purpose**: Handles CSV data and data manipulation
- **What it does**: Loads datasets, combines data, preprocesses text

#### **NumPy (Numerical Computing)**
- **Purpose**: Efficient array operations
- **What it does**: Supports pandas and scikit-learn operations

### 3.2 Frontend Technologies

- **HTML5**: Structure and content
- **CSS3**: Styling with gradients and animations
- **JavaScript**: Interactive functionality and API calls
- **Fetch API**: Communicates with Flask backend

---

## 4. Development Process

### 4.1 Step-by-Step Development

#### **Step 1: Project Planning**
We decided on:
- Machine learning approach (CountVectorizer + Logistic Regression)
- Web-based interface (Flask + HTML)
- Dataset structure (separate Fake.csv and True.csv files)

#### **Step 2: Dataset Preparation**
- Downloaded fake news datasets from Kaggle
- Split into two files: Fake.csv (fake news) and True.csv (real news)
- Each file contains columns like `title`, `text`, `subject`, `date`

#### **Step 3: Model Training Script**
Created `train_model.py` that:
1. Loads both CSV files
2. Combines and labels them (1=Fake, 0=Real)
3. Preprocesses text (lowercase, remove special chars)
4. Extracts features using CountVectorizer
5. Trains Logistic Regression model
6. Evaluates performance
7. Saves model and vectorizer as `.pkl` files

#### **Step 4: Flask Backend**
Created `app.py` that:
1. Loads the trained model
2. Provides a `/predict` endpoint for predictions
3. Preprocesses incoming headlines
4. Returns predictions with confidence scores

#### **Step 5: Web Dashboard**
Created `templates/index.html` that:
1. Provides a clean, modern interface
2. Takes user input (news headline)
3. Sends requests to Flask API
4. Displays predictions with visual feedback

#### **Step 6: Documentation & Setup**
- Created comprehensive README
- Added requirements.txt for dependencies
- Set up .gitignore to exclude large files
- Added LICENSE and CONTRIBUTING files

#### **Step 7: Git & GitHub**
- Initialized Git repository
- Committed all files
- Pushed to GitHub: https://github.com/Akhil-kukku/fake-news

---

## 5. Dataset Preparation

### 5.1 Dataset Structure

#### **Fake.csv**
Contains fake/misleading news articles with columns:
- `title`: The headline text
- `text`: Full article content
- `subject`: News category
- `date`: Publication date

#### **True.csv**
Contains verified real news articles with the same structure.

### 5.2 Data Processing Pipeline

```python
# 1. Load datasets
fake_df = pd.read_csv('Fake.csv')  # Load fake news
true_df = pd.read_csv('True.csv')  # Load real news

# 2. Add labels
fake_df['label'] = 1  # 1 = Fake
true_df['label'] = 0  # 0 = Real

# 3. Combine datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# 4. Shuffle data
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# 5. Extract text column (title or text)
X = df['title']  # or df['text']
y = df['label']
```

### 5.3 Text Preprocessing

```python
def preprocess_text(text):
    """Clean and normalize text"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text
```

**Why preprocessing?**
- Standardizes text format
- Removes noise (punctuation, numbers)
- Improves model performance
- Reduces feature space

---

## 6. Machine Learning Model Explained

### 6.1 CountVectorizer - Feature Extraction

#### What is CountVectorizer?

CountVectorizer converts text into numerical features by counting word occurrences.

#### Example:

**Input texts:**
- "Scientists discover new cure"
- "New technology breakthrough"

**CountVectorizer creates a matrix:**
```
              scientists  discover  new  cure  technology  breakthrough
Text 1             1         1       1    1       0           0
Text 2             0         0       1    0       1           1
```

#### Our Configuration:

```python
vectorizer = CountVectorizer(
    max_features=5000,      # Keep top 5000 most common words
    ngram_range=(1, 2),     # Use single words and word pairs
    stop_words='english',   # Remove common words (the, is, a)
    min_df=2                # Word must appear in at least 2 documents
)
```

**Parameters explained:**
- `max_features=5000`: Limits vocabulary to avoid overfitting
- `ngram_range=(1,2)`: Captures single words ("fake") and pairs ("fake news")
- `stop_words='english'`: Removes words like "the", "is", "a" (no predictive value)
- `min_df=2`: Ignores very rare words (likely typos or outliers)

### 6.2 Logistic Regression - Classification

#### What is Logistic Regression?

Despite the name, it's a **classification** algorithm (not regression). It predicts the probability that an input belongs to a particular class.

#### How it works:

1. **Training Phase:**
   - Learns weights for each feature (word)
   - Words common in fake news get positive weights
   - Words common in real news get negative weights

2. **Prediction Phase:**
   - Multiplies feature values by learned weights
   - Sums up the results
   - Applies sigmoid function to get probability (0-1)
   - If probability > 0.5 → Fake, else → Real

#### Our Configuration:

```python
model = LogisticRegression(
    max_iter=1000,    # Maximum training iterations
    random_state=42,  # For reproducibility
    C=1.0             # Regularization strength (prevents overfitting)
)
```

### 6.3 Training Process

```python
# 1. Split data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 2. Vectorize training data
X_train_vectorized = vectorizer.fit_transform(X_train)

# 3. Train model
model.fit(X_train_vectorized, y_train)

# 4. Vectorize test data (using same vocabulary)
X_test_vectorized = vectorizer.transform(X_test)

# 5. Make predictions
y_pred = model.predict(X_test_vectorized)

# 6. Evaluate
accuracy = accuracy_score(y_test, y_pred)
```

### 6.4 Model Evaluation Metrics

#### **Accuracy**
Percentage of correct predictions
```
Accuracy = (Correct Predictions) / (Total Predictions)
```

#### **Precision**
Of all predicted fake news, how many were actually fake?
```
Precision = True Positives / (True Positives + False Positives)
```

#### **Recall**
Of all actual fake news, how many did we catch?
```
Recall = True Positives / (True Positives + False Negatives)
```

#### **F1-Score**
Harmonic mean of precision and recall
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

### 6.5 Why This Approach?

**Advantages:**
✅ **Simple & Interpretable**: Easy to understand which words indicate fake news  
✅ **Fast Training**: Trains in seconds even on large datasets  
✅ **Low Resource**: Doesn't require GPU or extensive compute  
✅ **Good Baseline**: Often achieves 85-95% accuracy  
✅ **Explainable**: Can see which features (words) influence predictions  

**Limitations:**
❌ **Word Order**: Doesn't capture sequence (CountVectorizer treats text as bag-of-words)  
❌ **Context**: Misses semantic meaning and context  
❌ **Sarcasm**: Struggles with irony and sarcasm  
❌ **New Vocabulary**: Can't handle words not seen during training  

---

## 7. Code Walkthrough

### 7.1 train_model.py - Detailed Explanation

#### **Import Section**
```python
import pandas as pd              # Data manipulation
import numpy as np               # Numerical operations
from sklearn.model_selection import train_test_split  # Data splitting
from sklearn.feature_extraction.text import CountVectorizer  # Text to features
from sklearn.linear_model import LogisticRegression  # ML model
from sklearn.metrics import accuracy_score, classification_report  # Evaluation
import pickle                    # Model saving
import re                        # Text preprocessing
```

#### **Text Preprocessing Function**
```python
def preprocess_text(text):
    """
    Clean and normalize text data
    
    Args:
        text (str): Raw text input
        
    Returns:
        str: Cleaned text
    """
    if pd.isna(text):
        return ""
    
    # Convert to lowercase for consistency
    text = text.lower()
    
    # Remove special characters, numbers, punctuation
    # Keep only letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace (multiple spaces → single space)
    text = ' '.join(text.split())
    
    return text
```

**Why this preprocessing?**
- `lower()`: "FAKE" and "fake" should be treated the same
- `re.sub()`: Removes noise like "!!!" or "123" which don't help
- `split().join()`: Normalizes spacing for consistency

#### **Dataset Loading**
```python
# Load fake news dataset
fake_df = pd.read_csv('Fake.csv')
fake_df['label'] = 1  # Label as fake (1)

# Load real news dataset
true_df = pd.read_csv('True.csv')
true_df['label'] = 0  # Label as real (0)

# Combine both datasets
df = pd.concat([fake_df, true_df], ignore_index=True)

# Shuffle to mix fake and real news randomly
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
```

**Why shuffle?**
- Prevents model from learning patterns based on order
- Ensures training and test sets have both fake and real news
- `random_state=42`: Makes shuffle reproducible

#### **Column Detection**
```python
# Try to find the column containing headlines
title_columns = ['title', 'text', 'headline', 'content', 'article']
title_col = None

for col in title_columns:
    if col in df.columns:
        title_col = col
        break

# Rename to 'title' for consistency
df['title'] = df[title_col]
```

**Why?**
Different datasets use different column names. This makes the code flexible.

#### **Feature Extraction**
```python
# Preprocess all titles
df['cleaned_title'] = df['title'].apply(preprocess_text)

# Remove empty titles
df = df[df['cleaned_title'].str.len() > 0]

# Split into features (X) and labels (y)
X = df['cleaned_title']
y = df['label']

# Split into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # 20% for testing
    random_state=42,    # Reproducibility
    stratify=y          # Keep same ratio of fake/real in both sets
)

# Create and fit vectorizer
vectorizer = CountVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words='english',
    min_df=2
)

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
```

**Key points:**
- `stratify=y`: Ensures test set has same proportion of fake/real as training set
- `fit_transform()` on training: Learns vocabulary and transforms
- `transform()` on test: Uses learned vocabulary (no fitting!)

#### **Model Training**
```python
# Create model
model = LogisticRegression(
    max_iter=1000,    # Allow up to 1000 iterations to converge
    random_state=42,
    C=1.0             # Regularization (1.0 = moderate)
)

# Train model
model.fit(X_train_vectorized, y_train)

# Make predictions on test set
y_pred = model.predict(X_test_vectorized)

# Get probability scores
y_proba = model.predict_proba(X_test_vectorized)
```

#### **Model Evaluation**
```python
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2%}")

# Detailed report
print(classification_report(
    y_test, y_pred, 
    target_names=['Real News', 'Fake News']
))
```

**Classification report shows:**
- Precision: How many predicted fakes were actually fake?
- Recall: How many actual fakes did we find?
- F1-score: Balance between precision and recall
- Support: Number of samples for each class

#### **Model Saving**
```python
# Save trained model
with open('fake_news_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Save fitted vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
```

**Why save both?**
- Model: Contains learned weights for classification
- Vectorizer: Contains vocabulary and transformation logic
- Both needed to make predictions on new data

---

### 7.2 app.py - Detailed Explanation

#### **Import and Setup**
```python
from flask import Flask, render_template, request, jsonify
import pickle
import re

# Create Flask application
app = Flask(__name__)
```

#### **Model Loading**
```python
try:
    # Load trained model
    with open('fake_news_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    # Load fitted vectorizer
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    print("✓ Model and vectorizer loaded successfully")
except FileNotFoundError:
    print("❌ Model files not found. Run train_model.py first.")
    model = None
    vectorizer = None
```

**Error handling:**
- If model files don't exist, sets them to None
- API will return error message if user tries to predict

#### **Preprocessing Function**
```python
def preprocess_text(text):
    """Same preprocessing as training"""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text
```

**Critical:** Must match training preprocessing exactly!

#### **Home Route**
```python
@app.route('/')
def home():
    """Serve the dashboard HTML page"""
    return render_template('index.html')
```

**What it does:**
- When user visits `http://localhost:5000/`
- Flask serves `templates/index.html`

#### **Prediction Endpoint**
```python
@app.route('/predict', methods=['POST'])
def predict():
    """
    API endpoint for fake news prediction
    
    Expected JSON input:
    {
        "headline": "News headline text"
    }
    
    Returns JSON:
    {
        "prediction": "FAKE NEWS" or "REAL NEWS",
        "is_fake": true/false,
        "confidence": 87.52,
        "probabilities": {
            "real": 12.48,
            "fake": 87.52
        }
    }
    """
    try:
        # Check if model is loaded
        if model is None or vectorizer is None:
            return jsonify({
                'error': 'Model not loaded. Train model first.'
            }), 500
        
        # Get headline from request
        data = request.get_json()
        headline = data.get('headline', '')
        
        # Validate input
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
        
        # Vectorize (convert to same format as training data)
        headline_vectorized = vectorizer.transform([cleaned_headline])
        
        # Make prediction
        prediction = model.predict(headline_vectorized)[0]
        probability = model.predict_proba(headline_vectorized)[0]
        
        # Calculate confidence (maximum probability)
        confidence = max(probability) * 100
        
        # Format response
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
```

**Flow:**
1. Receive JSON with headline
2. Validate and preprocess
3. Vectorize using same vectorizer from training
4. Get prediction and probabilities
5. Format and return JSON response

#### **Health Check**
```python
@app.route('/health')
def health():
    """Check if API is running and model is loaded"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None and vectorizer is not None
    })
```

#### **Run Server**
```python
if __name__ == '__main__':
    print("\n" + "="*60)
    print("FAKE NEWS DETECTION API")
    print("="*60)
    print("Dashboard: http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
```

**Parameters:**
- `debug=True`: Auto-reload on code changes, detailed error messages
- `host='0.0.0.0'`: Accept connections from any IP
- `port=5000`: Run on port 5000

---

### 7.3 index.html - Detailed Explanation

#### **HTML Structure**
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fake News Detection Dashboard</title>
    <style>
        /* CSS styles here */
    </style>
</head>
<body>
    <!-- Dashboard content -->
    <script>
        /* JavaScript here */
    </script>
</body>
</html>
```

#### **Key CSS Features**

**Gradient Background:**
```css
body {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
}
```
Creates purple gradient background

**Card Container:**
```css
.container {
    background: white;
    border-radius: 20px;
    box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
    animation: fadeIn 0.5s ease-in;
}
```
White card with rounded corners and shadow

**Result Styling:**
```css
.result-fake {
    background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
    color: white;
}

.result-real {
    background: linear-gradient(135deg, #51cf66 0%, #37b24d 100%);
    color: white;
}
```
Red for fake, green for real

#### **JavaScript Functions**

**Main Analysis Function:**
```javascript
async function analyzeHeadline() {
    // Get input
    const headline = document.getElementById('headline').value.trim();
    
    // Validate
    if (!headline) {
        showError('Please enter a news headline');
        return;
    }
    
    // Show loading
    loading.classList.add('show');
    analyzeBtn.disabled = true;
    
    try {
        // Send API request
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ headline: headline })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Prediction failed');
        }
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        showError(error.message);
    } finally {
        // Hide loading
        loading.classList.remove('show');
        analyzeBtn.disabled = false;
    }
}
```

**Display Results:**
```javascript
function displayResults(data) {
    // Update text content
    resultHeader.textContent = data.prediction;
    confidence.textContent = `Confidence: ${data.confidence}%`;
    realProb.textContent = `${data.probabilities.real}%`;
    fakeProb.textContent = `${data.probabilities.fake}%`;
    
    // Apply appropriate styling
    resultSection.classList.remove('result-fake', 'result-real');
    if (data.is_fake) {
        resultSection.classList.add('result-fake');
    } else {
        resultSection.classList.add('result-real');
    }
    
    // Show result
    resultSection.classList.add('show');
}
```

**Example Headlines:**
```javascript
function useExample(element) {
    document.getElementById('headline').value = element.textContent.trim();
    document.getElementById('headline').focus();
}
```

---

## 8. API Documentation

### 8.1 Endpoints

#### **GET /**
Returns the web dashboard HTML page.

**Response:** HTML page

---

#### **POST /predict**
Predicts if a news headline is fake or real.

**Request Headers:**
```
Content-Type: application/json
```

**Request Body:**
```json
{
    "headline": "Scientists Discover Miracle Cure"
}
```

**Success Response (200):**
```json
{
    "prediction": "FAKE NEWS",
    "is_fake": true,
    "confidence": 87.52,
    "probabilities": {
        "real": 12.48,
        "fake": 87.52
    }
}
```

**Error Responses:**

*400 - Bad Request (empty headline)*
```json
{
    "error": "Please provide a headline"
}
```

*500 - Server Error (model not loaded)*
```json
{
    "error": "Model not loaded. Please train the model first."
}
```

---

#### **GET /health**
Health check endpoint to verify API status.

**Response:**
```json
{
    "status": "healthy",
    "model_loaded": true
}
```

---

### 8.2 Using the API with cURL

**Example 1: Make a prediction**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"headline": "Breaking: Scientists Discover Cure for Everything"}'
```

**Example 2: Check health**
```bash
curl http://localhost:5000/health
```

### 8.3 Using the API with Python

```python
import requests
import json

# API endpoint
url = "http://localhost:5000/predict"

# Headline to check
headline = "Scientists Make Shocking Discovery"

# Make request
response = requests.post(
    url,
    headers={"Content-Type": "application/json"},
    json={"headline": headline}
)

# Parse response
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']}%")
print(f"Probabilities: Real={result['probabilities']['real']}%, Fake={result['probabilities']['fake']}%")
```

### 8.4 Using the API with JavaScript

```javascript
async function checkHeadline(headline) {
    const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ headline: headline })
    });
    
    const data = await response.json();
    console.log(data);
    return data;
}

// Usage
checkHeadline("Breaking: World Leaders Meet").then(result => {
    console.log(`Result: ${result.prediction}`);
    console.log(`Confidence: ${result.confidence}%`);
});
```

---

## 9. Frontend Architecture

### 9.1 Design Principles

1. **Simplicity**: Clean, focused interface with minimal distractions
2. **Visual Feedback**: Colors and animations indicate status
3. **Responsiveness**: Works on all screen sizes
4. **Accessibility**: Clear labels and semantic HTML

### 9.2 User Flow

```
User visits homepage
    ↓
Enters news headline
    ↓
Clicks "Analyze Headline"
    ↓
Loading indicator appears
    ↓
JavaScript sends POST to /predict
    ↓
Flask processes and returns prediction
    ↓
Result displayed with:
    - Color coding (red/green)
    - Confidence percentage
    - Probability breakdown
```

### 9.3 UI Components

#### **Input Section**
- Large textarea for headline input
- Placeholder text with example
- Auto-focus on load

#### **Buttons**
- Primary button: "Analyze Headline" (gradient background)
- Secondary button: "Clear" (different color)
- Disabled state during processing

#### **Result Section**
- Large heading with prediction
- Confidence score
- Two boxes showing probabilities
- Animated slide-in effect

#### **Example Headlines**
- Clickable boxes
- Pre-filled examples
- Mix of likely fake and real

### 9.4 Responsive Design

**Desktop (>768px):**
- Wide container (800px max)
- Two-column probability display
- Large buttons

**Mobile (<768px):**
- Full-width container
- Stacked layout
- Touch-friendly buttons
- Adjusted font sizes

---

## 10. Deployment Guide

### 10.1 Local Development

**Requirements:**
- Python 3.8+
- pip (Python package manager)

**Steps:**
```bash
# 1. Clone repository
git clone https://github.com/Akhil-kukku/fake-news.git
cd fake-news

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add datasets (Fake.csv and True.csv)
# Download from Kaggle or your source

# 4. Train model
python train_model.py

# 5. Start server
python app.py

# 6. Open browser
# Visit http://localhost:5000
```

### 10.2 Production Deployment

#### **Option 1: Heroku**

1. Create `Procfile`:
```
web: gunicorn app:app
```

2. Add to `requirements.txt`:
```
gunicorn==21.2.0
```

3. Deploy:
```bash
heroku create fake-news-detector
git push heroku main
```

#### **Option 2: AWS EC2**

1. Launch EC2 instance (Ubuntu)
2. Install dependencies:
```bash
sudo apt update
sudo apt install python3-pip
pip3 install -r requirements.txt
```

3. Run with systemd or screen
4. Configure nginx as reverse proxy
5. Add SSL with Let's Encrypt

#### **Option 3: Docker**

Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "app.py"]
```

Build and run:
```bash
docker build -t fake-news-detector .
docker run -p 5000:5000 fake-news-detector
```

### 10.3 Performance Optimization

**Model Loading:**
- Load model once at startup (not per request)
- Use global variables for model and vectorizer
- Consider caching predictions for repeated headlines

**API Optimization:**
- Add rate limiting with Flask-Limiter
- Implement request validation
- Add CORS headers if needed
- Enable gzip compression

**Frontend Optimization:**
- Minify CSS and JavaScript
- Add loading states
- Implement error boundaries
- Cache static assets

### 10.4 Security Considerations

**Input Validation:**
- Sanitize user input
- Limit headline length
- Rate limit API requests

**Model Security:**
- Don't expose model files publicly
- Validate all inputs before prediction
- Log suspicious activity

**API Security:**
- Add authentication for production
- Use HTTPS only
- Implement CORS properly
- Add API keys for external access

---

## 11. Testing

### 11.1 Model Testing

**Test Cases:**
```python
test_headlines = [
    # Should be FAKE
    "Scientists Discover Miracle Cure That Doctors Hate",
    "You Won't Believe What Happens Next",
    "Shocking Truth About Celebrity Revealed",
    
    # Should be REAL
    "Government Announces New Budget Plan",
    "Research Team Publishes Study on Climate",
    "Stock Market Shows Moderate Gains Today"
]

for headline in test_headlines:
    prediction = predict(headline)
    print(f"{headline}\n  → {prediction}\n")
```

### 11.2 API Testing

**Using pytest:**
```python
import pytest
from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_page(client):
    response = client.get('/')
    assert response.status_code == 200

def test_predict_endpoint(client):
    response = client.post('/predict',
        json={'headline': 'Test headline'})
    assert response.status_code == 200
    data = response.get_json()
    assert 'prediction' in data
    assert 'confidence' in data

def test_empty_headline(client):
    response = client.post('/predict',
        json={'headline': ''})
    assert response.status_code == 400
```

---

## 12. Troubleshooting

### 12.1 Common Issues

**Issue: Model files not found**
```
Error: Model files not found. Please run train_model.py first.
```
**Solution:** Run `python train_model.py` to generate model files

---

**Issue: Port already in use**
```
Error: Address already in use
```
**Solution:** Change port in app.py or kill process on port 5000

---

**Issue: Low accuracy**
```
Model accuracy is below 70%
```
**Solutions:**
- Get larger, better quality dataset
- Increase max_features in CountVectorizer
- Try TF-IDF instead of CountVectorizer
- Use more sophisticated models (LSTM, BERT)

---

**Issue: Memory error**
```
MemoryError: Unable to allocate array
```
**Solutions:**
- Reduce max_features
- Use less data
- Increase system RAM
- Use sparse matrices

---

## 13. Future Enhancements

### 13.1 Model Improvements

1. **TF-IDF Vectorization**
   - Weights words by importance
   - Better than simple word counts

2. **Word Embeddings**
   - Use Word2Vec or GloVe
   - Captures semantic meaning

3. **Deep Learning**
   - LSTM networks for sequence
   - BERT for context understanding

4. **Ensemble Methods**
   - Combine multiple models
   - Random Forest + Logistic Regression

### 13.2 Feature Additions

1. **Full Article Analysis**
   - Not just headlines
   - Analyze complete text

2. **Source Credibility**
   - Check news source reputation
   - Cross-reference with fact-checkers

3. **Claim Extraction**
   - Identify specific claims
   - Verify each claim separately

4. **Explainability**
   - Show which words influenced decision
   - Highlight suspicious phrases

5. **User Feedback**
   - Allow users to report errors
   - Retrain model with corrections

### 13.3 UI Improvements

1. **Batch Processing**
   - Upload multiple headlines
   - Bulk analysis

2. **History Tracking**
   - Save previous checks
   - Export results

3. **Visualization**
   - Word clouds of fake vs real
   - Feature importance charts

4. **Mobile App**
   - Native iOS/Android apps
   - Share functionality

---

## 14. Learning Resources

### 14.1 NLP & Text Classification

- **Coursera**: Natural Language Processing Specialization
- **Fast.ai**: Practical Deep Learning for Coders
- **Stanford CS224N**: NLP with Deep Learning

### 14.2 Flask & Web Development

- **Flask Official Docs**: https://flask.palletsprojects.com/
- **Real Python**: Flask Mega-Tutorial
- **FreeCodeCamp**: Flask Course

### 14.3 Machine Learning

- **Coursera**: Machine Learning by Andrew Ng
- **Kaggle Learn**: Intro to Machine Learning
- **scikit-learn Documentation**: https://scikit-learn.org/

---

## 15. Conclusion

This fake news detection project demonstrates:

✅ **End-to-end ML pipeline**: From data loading to deployment  
✅ **Web application development**: Flask backend + HTML/CSS/JS frontend  
✅ **NLP techniques**: Text preprocessing and feature extraction  
✅ **Model training & evaluation**: Logistic Regression with proper metrics  
✅ **API design**: RESTful endpoints with JSON  
✅ **User interface**: Modern, responsive dashboard  
✅ **Documentation**: Comprehensive guides and comments  

### Key Takeaways

1. **Start Simple**: CountVectorizer + Logistic Regression is a great baseline
2. **Preprocess Carefully**: Same preprocessing for training and inference
3. **Evaluate Thoroughly**: Use multiple metrics, not just accuracy
4. **Document Everything**: Code comments and documentation help future you
5. **Iterate**: Start with MVP, then enhance based on feedback

### Next Steps

1. Deploy to production (Heroku, AWS, etc.)
2. Gather user feedback
3. Improve model with more data
4. Add advanced features
5. Share and contribute to open source

---

**Created by:** Akhil  
**Repository:** https://github.com/Akhil-kukku/fake-news  
**Date:** October 2025  

**Questions or suggestions?** Open an issue on GitHub!

---

*This documentation is part of the Fake News Detection Dashboard project, designed to be educational and accessible to developers of all skill levels.*
