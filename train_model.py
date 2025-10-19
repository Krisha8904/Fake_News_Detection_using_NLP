"""
Fake News Detection Model Training Script
Uses CountVectorizer + Logistic Regression to classify news headlines
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import re

def preprocess_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):
        return ""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def train_fake_news_detector():
    """Train the fake news detection model"""
    
    print("=" * 60)
    print("FAKE NEWS DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Load dataset
    print("\n[1/6] Loading dataset...")
    try:
        # Load Fake.csv and True.csv
        print("Loading Fake.csv...")
        fake_df = pd.read_csv('Fake.csv')
        print(f"  ✓ Fake news: {len(fake_df)} records")
        
        print("Loading True.csv...")
        true_df = pd.read_csv('True.csv')
        print(f"  ✓ Real news: {len(true_df)} records")
        
        # Add labels: 1 for fake, 0 for real
        fake_df['label'] = 1
        true_df['label'] = 0
        
        # Combine datasets
        df = pd.concat([fake_df, true_df], ignore_index=True)
        
        # Shuffle the dataset
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"✓ Combined dataset: {len(df)} records")
        print(f"  - Fake news: {len(fake_df)} ({len(fake_df)/len(df)*100:.1f}%)")
        print(f"  - Real news: {len(true_df)} ({len(true_df)/len(df)*100:.1f}%)")
        
        # Check what column contains the headlines
        print(f"\n  Available columns: {df.columns.tolist()}")
        
        # Common column names for headlines
        title_columns = ['title', 'text', 'headline', 'content', 'article']
        title_col = None
        
        for col in title_columns:
            if col in df.columns:
                title_col = col
                break
        
        if title_col is None:
            # Use the first text column
            for col in df.columns:
                if df[col].dtype == 'object' and col != 'label':
                    title_col = col
                    break
        
        if title_col is None:
            print("\n❌ Could not find a text column for headlines")
            return
        
        print(f"  Using column '{title_col}' for headlines")
        
        # Rename to 'title' for consistency
        if title_col != 'title':
            df['title'] = df[title_col]
        
    except FileNotFoundError as e:
        print(f"❌ Dataset files not found: {e}")
        print("Please ensure Fake.csv and True.csv are in the project directory")
        return
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Verify we have the required data
    if 'title' not in df.columns or 'label' not in df.columns:
        print("\n❌ Could not prepare dataset properly")
        return
    
    # Preprocess text
    print("\n[2/6] Preprocessing text data...")
    df['cleaned_title'] = df['title'].apply(preprocess_text)
    df = df[df['cleaned_title'].str.len() > 0]  # Remove empty titles
    print(f"✓ Text preprocessing complete")
    
    # Split data
    print("\n[3/6] Splitting data into train/test sets...")
    X = df['cleaned_title']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Testing set: {len(X_test)} samples")
    
    # Vectorization
    print("\n[4/6] Creating word frequency features with CountVectorizer...")
    vectorizer = CountVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # Use unigrams and bigrams
        stop_words='english',
        min_df=2
    )
    X_train_vectorized = vectorizer.fit_transform(X_train)
    X_test_vectorized = vectorizer.transform(X_test)
    print(f"✓ Feature matrix created: {X_train_vectorized.shape[1]} features")
    
    # Train model
    print("\n[5/6] Training Logistic Regression model...")
    model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    model.fit(X_train_vectorized, y_train)
    print("✓ Model training complete")
    
    # Evaluate
    print("\n[6/6] Evaluating model performance...")
    y_pred = model.predict(X_test_vectorized)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n{'='*60}")
    print(f"MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Real News', 'Fake News']))
    
    # Save model and vectorizer
    print("\n[7/7] Saving model and vectorizer...")
    with open('fake_news_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    print("✓ Model saved to: fake_news_model.pkl")
    print("✓ Vectorizer saved to: vectorizer.pkl")
    
    print(f"\n{'='*60}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*60}")
    print("\nModel Summary:")
    print(f"  • Algorithm: Logistic Regression")
    print(f"  • Features: Word frequency (CountVectorizer)")
    print(f"  • Training samples: {len(X_train)}")
    print(f"  • Test accuracy: {accuracy:.2%}")
    print(f"  • Total features: {X_train_vectorized.shape[1]}")

if __name__ == "__main__":
    train_fake_news_detector()
