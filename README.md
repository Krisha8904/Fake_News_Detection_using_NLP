# 🔍 Fake News Detection Dashboard

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

An AI-powered web application that classifies news headlines as **REAL** or **FAKE** using machine learning. Built with Flask and scikit-learn, this project demonstrates practical NLP techniques for fake news detection.

![Fake News Detection Dashboard](https://img.shields.io/badge/Status-Active-success)

## 🎯 Overview

This project uses **CountVectorizer** and **Logistic Regression** to detect patterns in fake news headlines based on word frequency features. The model analyzes the linguistic patterns and word usage to determine the authenticity of news headlines.

**Key Approach**: *"We trained a model on word frequency features to detect patterns in fake headlines"*

> 📚 **For detailed technical documentation, code explanations, and development guide, see [DOCUMENTATION.md](DOCUMENTATION.md)**

## 🛠️ Technology Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: 
  - CountVectorizer (Feature extraction)
  - Logistic Regression (Classification)
  - scikit-learn
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Data Processing**: Pandas, NumPy

## 📊 Model Details

**Algorithm**: Logistic Regression  
**Feature Engineering**: CountVectorizer with word frequency features  
**Approach**: "We trained a model on word frequency features to detect patterns in fake headlines"

### Features:
- Unigrams and bigrams (1-2 word combinations)
- Stop words removal
- Maximum 5000 features
- Text preprocessing (lowercase, special character removal)

## ✨ Features

- 🤖 **Machine Learning Model**: CountVectorizer + Logistic Regression
- 🎨 **Beautiful UI**: Modern, responsive web dashboard with gradient design
- 📊 **Confidence Scoring**: Get probability distributions for predictions
- ⚡ **Real-time Predictions**: Instant classification of news headlines
- 📈 **Model Metrics**: Detailed performance evaluation and statistics
- 🔄 **Easy Training**: Simple script to train on your own datasets
- 🌐 **REST API**: JSON API endpoints for integration

## 🖼️ Screenshots

### Dashboard Interface
The clean, modern interface allows users to input news headlines and get instant predictions with confidence scores.

### Prediction Results
- **Fake News Detection**: Red gradient with warning indicators
- **Real News Verification**: Green gradient with confirmation
- **Probability Distribution**: Visual breakdown of model confidence

## 📁 Project Structure

```
fake-news/
├── app.py                    # Flask API backend
├── train_model.py            # Model training script
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
├── .gitignore               # Git ignore rules
├── LICENSE                   # MIT License
├── templates/
│   └── index.html           # Web dashboard
├── Fake.csv                 # Fake news dataset
├── True.csv                 # Real news dataset
├── fake_news_model.pkl      # Trained model (generated after training)
└── vectorizer.pkl           # Fitted vectorizer (generated after training)
```

## 🚀 Setup Instructions

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Prepare Dataset

You can use your own dataset from Kaggle or let the script create a sample dataset.

**For custom dataset:**
- Download a fake news dataset from Kaggle (e.g., "Fake News Detection" dataset)
- Save it as `fake_news_data.csv` in the project directory
- Ensure it has columns: `title` (headline text) and `label` (0=Real, 1=Fake)

**For sample dataset:**
- The training script will automatically create a sample dataset if none exists

### 3. Train the Model

```powershell
python train_model.py
```

This will:
- Load or create the dataset
- Preprocess the text data
- Extract word frequency features
- Train the Logistic Regression model
- Evaluate performance
- Save the model and vectorizer

### 4. Start the Web Application

```powershell
python app.py
```

The dashboard will be available at: **http://localhost:5000**

## 🎮 Usage

1. Open your browser and navigate to `http://localhost:5000`
2. Enter a news headline in the text area
3. Click "🔍 Analyze Headline"
4. View the prediction results:
   - Classification (REAL NEWS or FAKE NEWS)
   - Confidence percentage
   - Probability distribution

### Example Headlines to Try:

**Likely Fake:**
- "Scientists Discover Cure for All Diseases Using This One Weird Trick"
- "Shocking: World Leaders Secretly Control Everything"

**Likely Real:**
- "Research Team Publishes Findings on Climate Change Impact"
- "Government Announces New Infrastructure Development Plan"

## 📈 Model Performance

After training, the script will display:
- Overall accuracy
- Classification report (precision, recall, F1-score)
- Confusion matrix
- Number of features used

## 🔧 Customization

### Adjust Model Parameters

Edit `train_model.py`:

```python
# CountVectorizer parameters
vectorizer = CountVectorizer(
    max_features=5000,      # Increase for more features
    ngram_range=(1, 2),     # Use (1, 3) for trigrams
    min_df=2                # Minimum document frequency
)

# Logistic Regression parameters
model = LogisticRegression(
    max_iter=1000,
    C=1.0                   # Regularization strength
)
```

### Change Port

Edit `app.py`:

```python
app.run(debug=True, host='0.0.0.0', port=5000)  # Change port here
```

## 📊 Dataset

This project uses two CSV files:
- **Fake.csv**: Collection of fake news articles
- **True.csv**: Collection of verified real news articles

### Dataset Sources
You can use datasets from:
1. **Kaggle - Fake News Detection** by Clement Bisaillon
2. **LIAR: A Benchmark Dataset for Fake News Detection**
3. **Getting Real about Fake News** (Kaggle Competition)
4. **Fake and Real News Dataset**

The training script automatically combines and labels the datasets.

## 🧪 API Endpoints

### `POST /predict`
Predict if a headline is fake or real.

**Request:**
```json
{
  "headline": "Your news headline here"
}
```

**Response:**
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

### `GET /health`
Check if the API and model are loaded.

## 🎨 Dashboard Features

- ✅ Real-time prediction
- ✅ Confidence scoring
- ✅ Probability visualization
- ✅ Example headlines
- ✅ Responsive design
- ✅ Error handling
- ✅ Loading animations

## 🐛 Troubleshooting

**Model files not found:**
- Run `python train_model.py` first to generate model files

**Dataset errors:**
- Ensure CSV has `title` and `label` columns
- Check for empty or null values

**Port already in use:**
- Change port in `app.py` or stop other Flask apps

## 🎓 Educational Purpose

This project demonstrates:
- Text preprocessing and feature extraction
- Machine learning classification techniques
- Flask web application development
- RESTful API design
- Interactive web dashboard creation
- Model evaluation and performance metrics

## 📝 Notes

- The model performs best with headlines similar to its training data
- Accuracy depends on dataset quality and size
- Consider using larger datasets for production use
- Model files (`.pkl`) are not included in the repository and must be generated

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contribution:
- Add more sophisticated NLP features (TF-IDF, word embeddings)
- Implement deep learning models (LSTM, BERT, Transformers)
- Add article content analysis (not just headlines)
- Create dataset upload functionality
- Add model retraining capability
- Implement user feedback collection
- Add more visualization features
- Improve UI/UX design

## 🔮 Future Enhancements

- [ ] TF-IDF vectorization option
- [ ] Deep learning models (LSTM, BERT)
- [ ] Full article text analysis
- [ ] Multi-language support
- [ ] User feedback system
- [ ] Model versioning
- [ ] Docker containerization
- [ ] Cloud deployment guide

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Akhil**

- GitHub: [@Akhil-kukku](https://github.com/Akhil-kukku)
- Repository: [fake-news](https://github.com/Akhil-kukku/fake-news)

## 🙏 Acknowledgments

- Dataset providers on Kaggle
- scikit-learn documentation and community
- Flask framework developers
- Open source community

## ⚠️ Disclaimer

This tool is for educational and research purposes only. While it can help identify potential fake news, it should not be the sole factor in determining the authenticity of news articles. Always verify information from multiple reliable sources.

---

**Built with ❤️ using Python, Flask, and scikit-learn**

If you find this project helpful, please consider giving it a ⭐!
